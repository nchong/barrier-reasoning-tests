/*
 * Blelloch exclusive prefix sum in OpenCL
 */

#define __1D_GRID
#define __1D_WORK_GROUP
#include "opencl.h"

// number of elements
#ifndef N
#error N must be defined
#endif

// control-plane
#ifndef dtype
#define dtype uint
#endif

// data-plane
#ifndef rtype
#define rtype uint
#endif

// specification header
#define __stringify_inner(x) #x
#define __stringify(x) __stringify_inner(x)
#define __spec_h(N) __concatenate(N, _spec.h)
#include __stringify(__spec_h(N))

__axiom(get_local_size(0) == N/2);
__axiom(get_num_groups(0) == 1);

__kernel void prescan(__local rtype *len) {
  __local rtype ghostsum[N];
  __local rtype result[N];

  dtype offset;
  dtype t = get_local_id(0);

#ifdef INC_UPSWEEP
  if (t < N/2) {
    result[2*t]   = len[2*t];
    result[2*t+1] = len[2*t+1];
  }

  offset = 1;
  for (
    dtype d=N/2;
    __invariant(upsweep_d_offset),
    __invariant(upsweep_barrier(tid,offset,result,len)),
    d > 0;
    d >>= 1) {
    __barrier_invariant(upsweep_barrier(tid,offset,result,len), tid, 2*tid, 2*tid+1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < d) {
      dtype ai = offset * (2 * t + 1) - 1;
      dtype bi = offset * (2 * t + 2) - 1;
      result[bi] = raddf_primed(result[ai], result[bi]);
    }
    offset <<= 1;
  }

  __assert(offset == N);
  __assert(upsweep_barrier(tid,/*offset=*/N,result,len));
#elif INC_DOWNSWEEP
  offset = N;
  __assume(upsweep_barrier(tid,/*offset=*/N,result,len));
  __array_snapshot(ghostsum, result);
  __assert(upsweep_barrier(tid,/*offset=*/N,ghostsum,len));
#endif

#ifdef INC_DOWNSWEEP
  if (t == 0) {
    result[N-1] = 0;
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
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < d) {
      dtype ai = offset * (2 * t + 1) - 1;
      dtype bi = offset * (2 * t + 2) - 1;
      rtype temp = result[ai];
      result[ai] = result[bi];
      result[bi] = raddf(result[bi], temp);
    }
  }
  __assert(offset == 1);
#elif INC_ENDSPEC
  __assume(upsweep_barrier(tid,/*offset=*/N,ghostsum,len));
  __assume(downsweep_barrier(tid,/*offset=*/0,result,ghostsum));
#endif

#ifdef INC_ENDSPEC
  // END SPECIFICATION
  __barrier_invariant(upsweep_barrier(tid,/*offset=*/N,ghostsum,len), upsweep_instantiation);
  __barrier_invariant(downsweep_barrier(tid,/*offset=*/0,result,ghostsum), tid, other_tid);
  barrier(CLK_LOCAL_MEM_FENCE);
  __assert(raddf(result[2*tid], len[2*tid]) == result[2*tid+1]);
  __assert(__implies(tid < other_tid, raddf(result[2*tid+1], len[2*tid+1]) <= result[2*other_tid]));
#endif
}
