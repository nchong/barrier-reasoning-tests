#include <cassert>
#include <cstdio>

#ifndef N
#error Must define number of elements N
#endif

#ifdef SCAN_BLELLOCH
  #include "blelloch.h"
#elif  SCAN_BRENTKUNG
  #include "brentkung.h"
#elif  SCAN_KOGGESTONE
  #include "koggestone.h"
#else
  #error Must define one of SCAN_BLELLOCH|SCAN_BRENTKUNG|SCAN_KOGGESTONE
#endif

#define __concatenate(x, y) x ## y
#define __stringify_inner(x) #x
#define __stringify(x) __stringify_inner(x)
#define __spec_h(N) __concatenate(N, _tests.h)
#include __stringify(__spec_h(N))

#define ASSERT_NO_CUDA_ERROR( callReturningErrorstatus ) {     \
  cudaError_t _err = callReturningErrorstatus;                  \
  if (_err != cudaSuccess) {                                    \
    fprintf(stderr,                                            \
            "Cuda error (%s/%d) in file '%s' in line %i\n",    \
            cudaGetErrorString(_err), _err, __FILE__, __LINE__); \
    exit(1);                                                   \
  }                                                            \
} while(0);

class Op1 {
  public:
  __device__ char operator() (char x, char y) const {
    if (y == 0) return x;
    if (x == 0 && y == 1) return 1;
    return 2;
  }
};

class Op2 {
  public:
  __device__ char operator() (char x, char y) const {
    if (y == 0) return x;
    if (y == 1) return 1;
    if (y == 2) return 2;
    return 0xff; //< should never fire
  }
};

__device__ void check(char *output, char *expected, bool *err) {
  unsigned tid = threadIdx.x;
#ifdef SCAN_KOGGESTONE
  if (output[tid] != expected[tid]) {
    *err = true;
  }
#else
  if ((output[2*tid] != expected[2*tid]) || (output[2*tid+1] != expected[2*tid+1])) {
    *err = true;
  }
#endif
}

/*
 * requires: err == false
 */
__global__ void wrapper(io_type *op1, io_type *op2, char *out, bool *err) {
  for (unsigned bid = blockIdx.x; bid<OP1_MEMBERS; bid+=gridDim.x) {
    char *input  = (char *)&(op1[bid].input);
    char *output = &(out[bid*N]);
    char *expected = (char *)&(op1[bid].output);
    scan(input, output, Op1());
    check(output, expected, err);
  }
  for (unsigned bid = blockIdx.x; bid<OP2_MEMBERS; bid+=gridDim.x) {
    char *input  = (char *)&(op2[bid].input);
    char *output = &(out[bid*N]);
    char *expected = (char *)&(op2[bid].output);
    scan(input, output, Op2());
    check(output, expected, err);
  }
}

int main(int argc, char **argv) {
  unsigned ntests = OP1_MEMBERS + OP2_MEMBERS;
  bool err = false;

  size_t op1_size = sizeof(io_type) * OP1_MEMBERS;
  size_t op2_size = sizeof(io_type) * OP2_MEMBERS;
  size_t out_size = sizeof(char) * N * max(OP1_MEMBERS, OP2_MEMBERS);
  size_t err_size = sizeof(bool);
  io_type *d_op1;
  io_type *d_op2;
  char    *d_out;
  bool    *d_err;
  ASSERT_NO_CUDA_ERROR(cudaMalloc((void **)&d_op1, op1_size));
  ASSERT_NO_CUDA_ERROR(cudaMalloc((void **)&d_op2, op2_size));
  ASSERT_NO_CUDA_ERROR(cudaMalloc((void **)&d_out, out_size));
  ASSERT_NO_CUDA_ERROR(cudaMalloc((void **)&d_err, err_size));

  ASSERT_NO_CUDA_ERROR(cudaMemcpy(d_op1,  op1, op1_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(cudaMemcpy(d_op2,  op2, op2_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(cudaMemcpy(d_err, &err, err_size, cudaMemcpyHostToDevice));

  ASSERT_NO_CUDA_ERROR(cudaDeviceSynchronize());
  cudaError_t cudaerr = cudaGetLastError();
  if (cudaerr != cudaSuccess) {
    printf("Pre-kernel error: %s.\n", cudaGetErrorString(cudaerr));
    return 1;
  }
  unsigned nblocks = min(max(OP1_MEMBERS, OP2_MEMBERS), 65535);
#ifdef SCAN_KOGGESTONE
  unsigned nthreads = N;
#else
  unsigned nthreads = N/2;
#endif
  wrapper<<<nblocks,nthreads>>>(d_op1, d_op2, d_out, d_err);
  ASSERT_NO_CUDA_ERROR(cudaDeviceSynchronize());
  cudaerr = cudaGetLastError();
  if (cudaerr != cudaSuccess) {
    printf("Post-kernel Error: %s\n", cudaGetErrorString(cudaerr));
    return 1;
  }

  ASSERT_NO_CUDA_ERROR(cudaMemcpy(&err, d_err, err_size, cudaMemcpyDeviceToHost));
  printf("nelements = %d; ntests = %d (%d/%d); result = %s\n", N, ntests, OP1_MEMBERS, OP2_MEMBERS, !err ? "pass" : "fail");

#if defined(PRINT_OUT1) || defined(PRINT_OUT2)
  char *out = (char *)malloc(out_size);
  assert(out);
  ASSERT_NO_CUDA_ERROR(cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost));
#if defined(PRINT_OUT1)
  for (unsigned i=0; i<OP1_MEMBERS; ++i) {
    for (unsigned j=0; j<N; ++j) {
      printf("%d %d %d %d %d\n", i, j, op1[i].input[j], out[i*N+j], op1[i].output[j]);
    //assert(out[i*N+j] == op1[i].output[j]);
    }
  }
#elif defined(PRINT_OUT2)
  for (unsigned i=0; i<OP2_MEMBERS; ++i) {
    for (unsigned j=0; j<N; ++j) {
      printf("%d %d %d %d %d\n", i, j, op2[i].input[j], out[i*N+j], op2[i].output[j]);
    //assert(out[i*N+j] == op2[i].output[j]);
    }
  }
#endif
  free(out);
#endif

  ASSERT_NO_CUDA_ERROR(cudaFree(d_op1));
  ASSERT_NO_CUDA_ERROR(cudaFree(d_op2));
  ASSERT_NO_CUDA_ERROR(cudaFree(d_out));
  ASSERT_NO_CUDA_ERROR(cudaFree(d_err));
  return 0;
}
