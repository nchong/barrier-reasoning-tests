#include <cassert>
#include <cstdio>

#ifndef N
#error N must be defined
#endif

#define PREDICATE(x) (((x & 1) == 0) ? 1 : 0)

__global__ void compact(uint *out, uint*in, uint *idx) {
  __shared__ uint flag[N];

  uint t = threadIdx.x;

  // (i) test each element with predicate p
  // flag = 1 if keeping element
  //        0 otherwise
  flag[t] = PREDICATE(in[t]);

  // (iii) scatter
  if (flag[t]) out[idx[t]] = in[t];
}

int main(int argc, char **argv) {
  // test data
  size_t ArraySize = N * sizeof(uint);
  uint *in  = (uint *)malloc(ArraySize);
  uint *out = (uint *)malloc(ArraySize);
  uint *idx = (uint *)malloc(ArraySize);
  klee_make_symbolic(in, ArraySize, "in");
  klee_make_symbolic(idx, ArraySize, "idx");

  // specification of prefix sum
  for (uint i=0; i<N; ++i) {
    for (uint j=i+1; j<N; ++j) {
      klee_assume(idx[i] + PREDICATE(in[i]) <= idx[j]);
    }
  }

  for (uint i=0; i<N; ++i) {
    assert(0 <= idx[i]);
    assert(idx[i] < N);
  }

  // create some memory objects on the device
  uint *d_in;
  uint *d_out;
  uint *d_idx;
  cudaMalloc((void **)&d_in, ArraySize);
  cudaMalloc((void **)&d_out, ArraySize);
  cudaMalloc((void **)&d_idx, ArraySize);

  // memcpy into these objects
  cudaMemcpy(d_in, in, ArraySize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx, idx, ArraySize, cudaMemcpyHostToDevice);

  // run the kernel
  compact<<<1,N>>>(d_out, d_in, d_idx);

  // memcpy back the result
  cudaMemcpy(out, d_out, ArraySize, cudaMemcpyDeviceToHost);

  // check results
  printf("TEST PASSED\n");

  // cleanup
  free(in);
  free(out);
  free(idx);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_idx);
  return 0;
}
