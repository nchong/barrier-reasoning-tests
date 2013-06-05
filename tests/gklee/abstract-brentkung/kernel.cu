#include <cassert>
#include <cstdio>

#ifndef N
#error N must be defined
#endif

#if rwidth == 8
  #define rtype uchar2
#elif rwidth == 16
  #define rtype ushort2
#elif rwidth == 32
  #define rtype uint2
#elif rwidth == 64
  #define rtype ulong2
#else
  #error rwidth must be defined
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

__global__ void brentkung(rtype *len, rtype *out, unsigned *error) {
  __shared__ rtype result[N];

  unsigned offset;
  unsigned t = threadIdx.x;

  if (t < N/2) {
    result[2*t].x   = 2*t;
    result[2*t].y   = 2*t+1;
    result[2*t+1].x = 2*t+1;
    result[2*t+1].y = 2*t+2;
  }

  offset = 1;
  for (unsigned d = N/2; d > 0; d /= 2) {
    __syncthreads();
    if (t < d) {
      unsigned ai = offset * (2 * t + 1) - 1;
      unsigned bi = offset * (2 * t + 2) - 1;
      if ( !((result[ai].x < result[ai].y) &&
             (               result[ai].y == result[bi].x) &&
             (                               result[bi].x < result[bi].y)) ) {
        *error = 1;
      }
      result[bi].x = result[ai].x;
    }
    offset *= 2;
  }

  for (unsigned d = 2; d < N; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (t < (d - 1)) {
      unsigned ai = (offset * (t + 1)) - 1;
      unsigned bi = ai + (offset >> 1);
      if ( !((result[ai].x < result[ai].y) &&
             (               result[ai].y == result[bi].x) &&
             (                               result[bi].x < result[bi].y)) ) {
        *error = 1;
      }
      result[bi].x = result[ai].x;
    }
  }
  __syncthreads();

  if (t < N/2) {
    out[2*t]   = result[2*t];
    out[2*t+1] = result[2*t+1];
  }
}

int main(int argc, char **argv) {
  // test data
  unsigned error;
  size_t ArraySize = N * sizeof(rtype);
  rtype *in  = (rtype *)malloc(ArraySize);
  rtype *out = (rtype *)malloc(ArraySize);
#ifdef _SYM
  klee_make_symbolic(in, ArraySize, "in");
  klee_make_symbolic(&error, sizeof(unsigned), "error");
  klee_assume(error == 0);
#else
  error = 0;
#endif

  // create arrays on device
  unsigned *d_error;
  rtype *d_in;
  rtype *d_out;
  ASSERT_NO_CUDA_ERROR(cudaMalloc((void **)&d_error, sizeof(unsigned)));
  ASSERT_NO_CUDA_ERROR(cudaMalloc((void **)&d_in, ArraySize));
  ASSERT_NO_CUDA_ERROR(cudaMalloc((void **)&d_out, ArraySize));

  // memcpy into arrays
  ASSERT_NO_CUDA_ERROR(cudaMemcpy(d_error, &error, sizeof(unsigned), cudaMemcpyHostToDevice));
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
  brentkung<<<1,(N/2)>>>(d_in, d_out, d_error);
#ifndef _SYM
  ASSERT_NO_CUDA_ERROR(cudaDeviceSynchronize());
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Post-kernel Error: %s\n", cudaGetErrorString(err));
    return 1;
  }
#endif

  // memcpy back the result
  ASSERT_NO_CUDA_ERROR(cudaMemcpy(&error, d_error, sizeof(unsigned), cudaMemcpyDeviceToHost));
  ASSERT_NO_CUDA_ERROR(cudaMemcpy(out, d_out, ArraySize, cudaMemcpyDeviceToHost));

  // check monotonic specification
#ifdef _SYM
  unsigned i,j;
  klee_make_symbolic(&i, sizeof(unsigned), "i");
  klee_make_symbolic(&j, sizeof(unsigned), "j");
  klee_assume(i < N);
  klee_assume(j < N);
  klee_assume(i < j);
  if (error) {
    printf("TEST FAIL: ASSERTION FIRED\n");
    assert(false);
  }
  if (!( out[i].x == 0 && out[i].y == i+1 && out[i].y < out[j].y )) {
    printf("TEST FAIL: MONOTONIC SPECIFICATION\n");
    assert(false);
  }
#else
  // check full specification
  assert(error == 0);
  for (unsigned i=0; i<N; ++i) {
    printf("out[%d] = (%d,%d) (0,%d)\n", i, out[i].x, out[i].y, i+1);
    assert(out[i].x == 0);
    assert(out[i].y == i+1);
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
