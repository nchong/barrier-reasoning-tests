#ifdef MAIN
#include <cassert>
#include <cstdio>
#endif

#include "cuda.h"

#ifndef N
#error N must be defined
#endif

// control-plane
#if dwidth == 8
  #define dtype unsigned char
#elif dwidth == 16
  #define dtype unsigned short
#elif dwidth == 32
  #define dtype unsigned int
#elif dwidth == 64
  #define dtype unsigned long
#else
  #error dwidth must be defined
#endif

// data-plane
#if rwidth == 8
  #define rtype unsigned char
  #define nooverflow_add(x,y) __add_noovfl_unsigned_char(x,y)
#elif rwidth == 16
  #define rtype unsigned short
  #define nooverflow_add(x,y) __add_noovfl_unsigned_short(x,y)
#elif rwidth == 32
  #define rtype unsigned int
  #define nooverflow_add(x,y) __add_noovfl_unsigned_int(x,y)
#elif rwidth == 64
  #define rtype unsigned long
  #define nooverflow_add(x,y) __add_noovfl_unsigned_long(x,y)
#else
  #error rwidth must be defined
#endif

// specification header
#define __stringify_inner(x) #x
#define __stringify(x) __stringify_inner(x)
#define __spec_h(N) __concatenate(N, _spec.h)
#include __stringify(__spec_h(N))

__global__ void blelloch(rtype *len, rtype *out) {
  __shared__ rtype ghostsum[N];
  __shared__ rtype result[N];

  dtype offset;
  dtype t = threadIdx.x;

#ifdef INC_UPSWEEP
  if (t < N/2) {
    result[2*t]   = len[2*t];
    result[2*t+1] = len[2*t+1];
  }

  offset = 1;
  for (
    dtype d = N/2;
    __invariant(upsweep_d_offset),
    __invariant(upsweep_barrier(tid,offset,result,len)),
    d > 0;
    d /= 2) {
    __barrier_invariant(upsweep_barrier(tid,offset,result,len), tid, 2*tid, 2*tid+1);
    __syncthreads();
    if (t < d) {
      dtype ai = offset * (2 * t + 1) - 1;
      dtype bi = offset * (2 * t + 2) - 1;
#if defined(INC_ENDSPEC) && defined(BINOP_ADD)
      result[bi] = nooverflow_add(result[ai], result[bi]);
#else
      result[bi] = raddf_primed(result[ai], result[bi]);
#endif
    }
    offset *= 2;
  }
#elif INC_DOWNSWEEP
  offset = N;
  __assume(upsweep_barrier(tid,/*offset=*/N,result,len));
//__non_temporal(__assert(upsweep_barrier(tid,/*offset=*/N,ghostsum,len)));
#endif

#ifdef INC_DOWNSWEEP
  __array_snapshot(ghostsum, result);

  if (t == 0) {
    result[N-1] = ridentity;
  }

  for (
    dtype d = 1;
    __invariant(downsweep_d_offset),
    __invariant(upsweep_barrier(tid,/*offset=*/N,ghostsum,len)),
    __invariant(downsweep_barrier(tid,div2(offset),result,ghostsum)),
    d < N;
    d <<= 1) {
    offset >>= 1;
    __barrier_invariant(upsweep_barrier(tid,/*offset=*/N,ghostsum,len), tid);
    __barrier_invariant(downsweep_barrier(tid,offset,result,ghostsum), tid, div2(tid));
    __syncthreads();
    if (t < d) {
      dtype ai = offset * (2 * t + 1) - 1;
      dtype bi = offset * (2 * t + 2) - 1;
      rtype temp = result[ai];
      result[ai] = result[bi];
#if defined(INC_ENDSPEC) && defined(BINOP_ADD)
      result[bi] = nooverflow_add(result[bi], temp);
#else
      result[bi] = raddf(result[bi], temp);
#endif
    }
  }
#elif INC_ENDSPEC
  __assume(upsweep_barrier(tid,/*offset=*/N,ghostsum,len));
  __assume(downsweep_barrier(tid,/*offset=*/0,result,ghostsum));
#endif

#ifdef INC_ENDSPEC
#if defined(SPEC_THREADWISE)
  __barrier_invariant(final_upsweep_barrier(tid,ghostsum,len), upsweep_instantiation);
  __barrier_invariant(final_downsweep_barrier(tid,result,ghostsum), tid, other_tid);
  __syncthreads();
  __non_temporal(__assert(raddf(result[2*tid], len[2*tid]) == result[2*tid+1]));
  __non_temporal(__assert(__implies(tid < other_tid, raddf(result[2*tid+1], len[2*tid+1]) <= result[2*other_tid])));
#elif defined(SPEC_ELEMENTWISE)
  __barrier_invariant(final_upsweep_barrier(tid,ghostsum,len), upsweep_instantiation);
  __barrier_invariant(final_downsweep_barrier(tid,result,ghostsum), x2t(tid), x2t(other_tid));
  __syncthreads();
  __non_temporal(__assert(__implies(tid < other_tid, raddf(result[tid], len[tid]) <= result[other_tid])));
#else
  #error SPEC_THREADWISE|SPEC_ELEMENTWISE must be defined
#endif

  if (t < N/2) {
    out[2*t]   = result[2*t];
    out[2*t+1] = result[2*t+1];
  }
#endif


#ifdef FORCE_FAIL
  __assert(false);
#endif
}

#ifdef MAIN
int main(int argc, char **argv) {
  // test data
  size_t ArraySize = N * sizeof(rtype);
  rtype *in  = (rtype *)malloc(ArraySize);
  rtype *out = (rtype *)malloc(ArraySize);
#ifdef _SYM
  klee_make_symbolic(in, ArraySize, "in");
#else
  for (dtype i=0; i<N; ++i) {
    in[i] = i+1;
  }
#endif

  // create arrays on device
  rtype *d_in;
  rtype *d_out;
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
  for (dtype i=0; i<N; ++i) {
    printf("out[%d] = %d\n", i, out[i]);
  }
#endif
  for (dtype i=0; i<(N-1); ++i) {
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
#endif
