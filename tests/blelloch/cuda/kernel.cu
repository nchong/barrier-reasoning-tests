#include <cassert>
#include <cstdio>

#ifndef N
#error N must be defined
#endif

__global__ void blelloch(unsigned *len, unsigned *out) {
  __shared__ unsigned result[N];

  unsigned offset;
  unsigned t = threadIdx.x;

  if (t < N/2) {
    result[2*t]   = len[2*t];
    result[2*t+1] = len[2*t+1];
  }

  offset = 1;
  for (unsigned d = N/2; d > 0; d /= 2) {
    __syncthreads();
    if (t < d) {
      unsigned ai = offset * (2 * t + 1) - 1;
      unsigned bi = offset * (2 * t + 2) - 1;
      result[bi] += result[ai];
    }
    offset *= 2;
  }

  if (t == 0) {
    result[N-1] = 0;
  }

  for (unsigned d = 1; d < N; d *= 2) {
    offset /= 2;
    __syncthreads();
    if (t < d) {
      unsigned ai = offset * (2 * t + 1) - 1;
      unsigned bi = offset * (2 * t + 2) - 1;
      unsigned temp = result[ai];
      result[ai] = result[bi];
      result[bi] += temp;
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
  size_t ArraySize = N * sizeof(unsigned);
  unsigned *in  = (unsigned *)malloc(ArraySize);
  unsigned *out = (unsigned *)malloc(ArraySize);
#ifdef _SYM
  klee_make_symbolic(in, ArraySize, "in");
#else
  for (unsigned i=0; i<N; ++i) {
    in[i] = i+1;
  }
#endif

  // create arrays on device
  unsigned *d_in;
  unsigned *d_out;
  cudaMalloc((void **)&d_in, ArraySize);
  cudaMalloc((void **)&d_out, ArraySize);

  // memcpy into arrays
  cudaMemcpy(d_in, in, ArraySize, cudaMemcpyHostToDevice);

  // run the kernel
  blelloch<<<1,(N/2)>>>(d_in, d_out);

  // memcpy back the result
  cudaMemcpy(out, d_out, ArraySize, cudaMemcpyDeviceToHost);

  // check specification
#ifndef _SYM
  for (unsigned i=0; i<N; ++i) {
    printf("out[%d] = %d\n", i, out[i]);
  }
#endif
  for (unsigned i=0; i<(N-1); ++i) {
    assert(out[i] + in[i] <= out[i+1] && "SPEC FAILED");
  }
  printf("TEST PASSED\n");

  // cleanup
  free(in);
  free(out);
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
