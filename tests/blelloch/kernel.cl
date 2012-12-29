#define __1D_GRID
#define __1D_WORK_GROUP
#include "opencl.h"

#define raddf(x,y) (x + y)

#define DTYPE unsigned
#define RTYPE unsigned char

__axiom(get_local_size(0) == N/2);
__axiom(get_num_groups(0) == 1);

__kernel void prescan(__local RTYPE *len) {
  __local RTYPE ghostsum[N];
  __local RTYPE result[N];

  DTYPE t = get_local_id(0);

  if (t < N/2) {
    result[2*t]   = len[2*t];
    result[2*t+1] = len[2*t+1];
  }

  DTYPE offset = 1;
  for (
    DTYPE d=N/2;
    __invariant(upsweep_d_offset),
    __invariant(upsweep_barrier(tid,offset,result,len)),
    d > 0;
    d >>= 1) {
    __barrier_invariant(upsweep_barrier(tid,offset,result,len), tid, 2*tid, 2*tid+1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < d) {
      DTYPE ai = offset * (2 * t + 1) - 1;
      DTYPE bi = offset * (2 * t + 2) - 1;
      result[bi] += result[ai];
    }
    offset <<= 1;
  }

  __assert(offset == N);
  __assert(upsweep_barrier(tid,/*offset=*/N,result,len)),
  __array_snapshot(ghostsum, result);
  __assert(upsweep_barrier(tid,/*offset=*/N,ghostsum,len));

  if (t == 0) {
    result[N-1] = 0;
  }

  for (
    DTYPE d = 1;
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
      DTYPE ai = offset * (2 * t + 1) - 1;
      DTYPE bi = offset * (2 * t + 2) - 1;
      RTYPE temp = result[ai];
      result[ai] = result[bi];
      result[bi] += temp;
    }
  }
  __assert(offset == 1);

#if 0
  // END SPECIFICATION
  __barrier_invariant(upsweep_barrier(tid,/*offset=*/N,ghostsum,len), upsweep_instantiation);
  __barrier_invariant(downsweep_barrier(tid,/*offset=*/0,result,ghostsum), tid, other_tid);
  barrier(CLK_LOCAL_MEM_FENCE);
  __assert(result[2*tid] + len[2*tid] == result[2*tid+1]);
  __assert(__implies(tid < other_tid, result[2*tid+1] + len[2*tid+1] <= result[2*other_tid]));
#endif
}
