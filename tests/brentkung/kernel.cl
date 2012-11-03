#define __1D_GRID
#define __1D_WORK_GROUP
#include "opencl.h"

__axiom(get_local_size(0) == N/2);
__axiom(get_num_groups(0) == 1);

__kernel void scan(__local unsigned *len) {
  __local unsigned ghostsum[N];
  __local unsigned result[N];

  unsigned t = get_local_id(0);

  if (t < N/2) {
    result[2*t]   = len[2*t];
    result[2*t+1] = len[2*t+1];
  }

  unsigned offset = 1;
  for (
    unsigned d=N/2;
    __invariant(upsweep_d_offset),
    __invariant(upsweep_barrier(tid,offset,result,len)),
    d > 0;
    d >>= 1) {
    __barrier_invariant(upsweep_barrier(tid,offset,result,len), tid, 2*tid, 2*tid+1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < d) {
      unsigned ai = offset * (2 * t + 1) - 1;
      unsigned bi = offset * (2 * t + 2) - 1;
      result[bi] += result[ai];
    }
    offset <<= 1;
  }

  __assert(offset == N);
  __assert(upsweep_barrier(tid,/*offset=*/N,result,len)),
  __array_snapshot(ghostsum, result);
  __assert(upsweep_barrier(tid,/*offset=*/N,ghostsum,len));

  for (
    unsigned d = 2;
    __invariant(downsweep_d_offset),
    __invariant(upsweep_barrier(tid,/*offset=*/N,ghostsum,len)),
    __invariant(downsweep_barrier(tid,offset,result,ghostsum)),
    d < N;
    d <<= 1) {
    offset >>= 1;
    __barrier_invariant(upsweep_barrier(tid,/*offset=*/N,ghostsum,len), tid);
    __barrier_invariant(downsweep_barrier(tid,mul2(offset),result,ghostsum), tid, lf_ai_tid(tid), lf_bi_tid(tid)),
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < (d - 1)) {
      unsigned ai = (offset * (t + 1)) - 1;
      unsigned bi = ai + (offset >> 1);
      result[bi] += result[ai];
    }
  }
  __assert(offset == 2);
}
