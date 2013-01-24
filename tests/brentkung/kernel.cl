/*
 * Brent-Kung inclusive prefix sum in OpenCL
 */

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

__kernel void scan(__local unsigned *len) {
  __local unsigned ghostsum[N];
  __local unsigned result[N];

  unsigned offset;
  unsigned t = get_local_id(0);

#ifdef INC_UPSWEEP
  if (t < N/2) {
    result[2*t]   = len[2*t];
    result[2*t+1] = len[2*t+1];
  }

  offset = 1;
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
#elif INC_DOWNSWEEP
  offset = N;
  __assume(upsweep_barrier(tid,/*offset=*/N,result,len));
#endif

#ifdef INC_DOWNSWEEP
  __array_snapshot(ghostsum, result);

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
      result[bi] = raddf(result[ai], result[bi]);
    }
  }
#elif INC_ENDSPEC
//__non_temporal(__assume(upsweep_barrier(tid,/*offset=*/N,ghostsum,len)));
//__non_temporal(__assume(downsweep_barrier(tid,/*offset=*/2,result,ghostsum)));
  __assume(upsweep_barrier(tid,/*offset=*/N,ghostsum,len));
  __assume(downsweep_barrier(tid,/*offset=*/2,result,ghostsum));
#endif

#ifdef INC_ENDSPEC
  __barrier_invariant(final_upsweep_barrier(tid,ghostsum,len), upsweep_instantiation);
  __barrier_invariant(final_downsweep_barrier(tid,result,ghostsum), downsweep_instantiation);
  barrier(CLK_LOCAL_MEM_FENCE);
#if defined(SPEC_THREADWISE)
  __non_temporal(__assert(__implies(tid == 0, raddf(result[0], len[1]) == result[1])));
  __non_temporal(__assert(__implies(tid  < ((N/2)-1), raddf(result[mul2(tid)+1], len[mul2(tid)+2]) == result[mul2(tid)+2])));
  __non_temporal(__assert(__implies((tid < other_tid) & (other_tid < ((N/2)-1)), raddf(result[mul2(tid)+2], len[mul2(tid)+3]) <= result[mul2(other_tid)+1])));
  __non_temporal(__assert(__implies((0 < tid) & (tid < ((N/2)-1)) & (other_tid == 0), raddf(result[mul2(tid)+2],len[mul2(tid)+3]) <= result[N-1])));
#elif defined(SPEC_ELEMENTWISE)
  __non_temporal(__assert(__implies(tid < other_tid, raddf(result[tid], len[tid]) <= result[other_tid])));
#else
  #error SPEC_THREADWISE|SPEC_ELEMENTWISE must be defined
#endif
#endif

#ifdef FORCE_FAIL
  __assert(false);
#endif
}
