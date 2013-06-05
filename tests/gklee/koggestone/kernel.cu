#include <cassert>
#include <cstdio>

#ifndef N
#error N must be defined
#endif

#if rwidth == 8
  #define rtype unsigned char
  #define MAX_RTYPE 0xff
#elif rwidth == 16
  #define rtype unsigned short
  #define MAX_RTYPE 0xffff
#elif rwidth == 32
  #define rtype unsigned int
  #define MAX_RTYPE 0xffffffff
#elif rwidth == 64
  #define rtype unsigned long
  #define MAX_RTYPE 0xffffffffffffffff
#else
  #error rwidth must be defined
#endif

#ifdef BINOP_ADD
  #define OP(x,y) (x <= MAX_RTYPE - y ? (x + y) : MAX_RTYPE)
#elif BINOP_OR
  #define OP(x,y) (x | y)
#elif BINOP_MAX
  #define OP(x,y) (x < y ? y : x)
#else
  #error Must define one of BINOP_ADD|BINOP_OR|BINOP_MAX
#endif

#ifdef _SYM
  #define ASSERT_NO_CUDA_ERROR( callReturningErrorstatus ) callReturningErrorstatus
#else
  #define ASSERT_NO_CUDA_ERROR( callReturningErrorstatus ) {     \
    cudaError_t err = callReturningErrorstatus;                  \
    if (err != cudaSuccess) {                                    \
      fprintf(stderr,                                            \
              "Cuda error (%s/%d) in file '%s' in line %i\n",    \
              cudaGetErrorString(err), err, __FILE__, __LINE__); \
      exit(1);                                                   \
    }                                                            \
  } while(0);
#endif

__global__ void koggestone(rtype *len, rtype *out) {
  __shared__ rtype result[N];

  unsigned t = threadIdx.x;

  result[t] = len[t];

  __syncthreads();

  rtype temp;
  for (unsigned offset = 1; offset < N; offset *= 2) {
    if (t >= offset) {
      temp = result[t-offset];
    }
    __syncthreads();
    if (t >= offset) {
      result[t] = OP(result[t], temp);
    }
    __syncthreads();
  }

  out[t] = result[t];
}

int main(int argc, char **argv) {
  // test data
  size_t ArraySize = N * sizeof(rtype);
  rtype *in  = (rtype *)malloc(ArraySize);
  rtype *out = (rtype *)malloc(ArraySize);
#ifdef _SYM
  klee_make_symbolic(in, ArraySize, "in");
#else
  for (unsigned i=0; i<N; ++i) {
    in[i] = 101+i;
  }
#endif

  // create arrays on device
  rtype *d_in;
  rtype *d_out;
  ASSERT_NO_CUDA_ERROR(cudaMalloc((void **)&d_in, ArraySize));
  ASSERT_NO_CUDA_ERROR(cudaMalloc((void **)&d_out, ArraySize));

  // memcpy into arrays
  ASSERT_NO_CUDA_ERROR(cudaMemcpy(d_in, in, ArraySize, cudaMemcpyHostToDevice));

  // run the kernel
  ASSERT_NO_CUDA_ERROR(cudaDeviceSynchronize());
#ifndef _SYM
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pre-kernel error: %s.\n", cudaGetErrorString(err));
    return 1;
  }
#endif
  koggestone<<<1,N>>>(d_in, d_out);
#ifndef _SYM
  ASSERT_NO_CUDA_ERROR(cudaDeviceSynchronize());
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Post-kernel Error: %s\n", cudaGetErrorString(err));
    return 1;
  }
#endif

  // memcpy back the result
  ASSERT_NO_CUDA_ERROR(cudaMemcpy(out, d_out, ArraySize, cudaMemcpyDeviceToHost));

  // check monotonic specification
#ifdef _SYM
  unsigned i,j;
  klee_make_symbolic(&i, sizeof(unsigned), "i");
  klee_make_symbolic(&j, sizeof(unsigned), "j");
  klee_assume(i < N);
  klee_assume(j < N);
  klee_assume(i < j);
  if (!( OP(out[i], in[i+1]) <= out[j] )) {
    printf("TEST FAIL: MONOTONIC SPECIFICATION\n");
    assert(false);
  }
#else
  // check full specification
  rtype sum = in[0];
  for (unsigned i=0; i<N; ++i) {
    printf("out[%d] = %d (%d)\n", i, out[i], sum);
    assert(sum == out[i]);
    if (i < N-1) sum = OP(sum,in[i+1]);
  }
#endif
  printf("TEST PASSED\n");

  // cleanup
  free(in);
  free(out);
  ASSERT_NO_CUDA_ERROR(cudaFree(d_in));
  ASSERT_NO_CUDA_ERROR(cudaFree(d_out));
  return 0;
}
