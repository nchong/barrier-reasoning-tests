/*
 * Brent-Kung inclusive prefix sum in OpenCL
 */

// number of elements
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

#define __1D_WORK_GROUP
#define __1D_GRID
#include "opencl.h"

DECLARE_UF_BINARY(A, rtype, rtype, rtype);
DECLARE_UF_BINARY(A1, rtype, rtype, rtype);

// specification header
#define __stringify_inner(x) #x
#define __stringify(x) __stringify_inner(x)
#define __spec_h(N) __concatenate(N, _spec.h)
#include __stringify(__spec_h(N))

__kernel void scan(__local rtype *len) {
#ifdef BINOP_PAIR
  __local uint2 ghostsum[N];
  __local uint2 result[N];
#else
  __local rtype ghostsum[N];
  __local rtype result[N];
#endif

  dtype offset;
  dtype t = get_local_id(0);

#ifdef INC_UPSWEEP
  if (t < N/2) {
#ifdef BINOP_PAIR
    result[2*t]   = (uint2)(2*t  , 2*t+1);
    result[2*t+1] = (uint2)(2*t+1, 2*t+2);
#else
    result[2*t]   = len[2*t];
    result[2*t+1] = len[2*t+1];
#endif
  }

  offset = 1;
  for (
    dtype d=N/2;
    __invariant(upsweep_d_offset),
//  __invariant(__uniform_int(offset)),
//  __invariant(__uniform_int(d)),
//  __invariant(__uniform_bool(__enabled())),
#ifdef CHECK_RACE
    __invariant(__no_write(len)),
    __invariant(
      __read_implies(result,
        (offset > 1) &
        (__read_offset(result)/sizeof(rtype) == ai_idx(div2(offset),tid) |
         __read_offset(result)/sizeof(rtype) == bi_idx(div2(offset),tid)))
    ),
    __invariant(
      __write_implies(result,
        __ite(offset == 1,
          __write_offset(result)/sizeof(rtype) == ai_idx(1,tid) |
          __write_offset(result)/sizeof(rtype) == bi_idx(1,tid),
          __write_offset(result)/sizeof(rtype) == bi_idx(div2(offset),tid)))
    ),
    __invariant(__implies(__read(result) & (offset == N), tid == 0)),
    __invariant(__implies(__write(result) & (offset == N), tid == 0)),
#endif
#ifdef CHECK_BI
    __invariant(upsweep_barrier(tid,offset,result,len)),
#endif
    d > 0;
    d >>= 1) {
#ifdef CHECK_BI_ACCESS
    upsweep_barrier_permissions(tid,offset,result,len)
#endif
#ifdef CHECK_BI
    __barrier_invariant(upsweep_barrier(tid,offset,result,len), tid, 2*tid, 2*tid+1);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < d) {
      dtype ai = offset * (2 * t + 1) - 1;
      dtype bi = offset * (2 * t + 2) - 1;
#ifdef CHECK_BI
#ifdef BINOP_PAIR
      __assert(result[ai].lo < result[ai].hi);
      __assert(                result[ai].hi == result[bi].lo);
      __assert(                                 result[bi].lo < result[bi].hi);
#endif
#endif
#if defined(BINOP_PAIR)
      result[bi].lo = result[ai].lo;
#elif defined(INC_ENDSPEC) && defined(BINOP_ADD)
      result[bi] = nooverflow_add(result[ai], result[bi]);
#else
      result[bi] = raddf_primed(result[ai], result[bi]);
#endif
    }
    offset <<= 1;
  }

//__assert(offset == N);
//__non_temporal(__assert(upsweep_barrier(tid,/*offset=*/N,result,len)));
#elif INC_DOWNSWEEP
  offset = N;
  __non_temporal(__assume(upsweep_barrier(tid,/*offset=*/N,result,len)));
//__non_temporal(__assert(upsweep_barrier(tid,/*offset=*/N,ghostsum,len)));
#endif

#ifdef INC_DOWNSWEEP
#ifdef CHECK_BI
  __array_snapshot(ghostsum, result);
#endif

  for (
    dtype d = 2;
    __invariant(downsweep_d_offset),
//  __invariant(__uniform_int(offset)),
//  __invariant(__uniform_int(d)),
//  __invariant(__uniform_bool(__enabled())),
#ifdef CHECK_RACE
    __invariant(__no_write(len)),
    __invariant(__implies(__write(result) & (__write_offset(result)/sizeof(rtype) == (N-1)), tid == 0)),
    __invariant(__implies(__read(result) & (offset == N), tid == 0)),
    __invariant(__implies(__write(result) & (offset == N), 
        (tid == 0) & (__write_offset(result)/sizeof(rtype) == (N-1)))
    ),
    __invariant(__implies(__read(result) & (offset < N),
      __read_offset(result)/sizeof(rtype) == lf_ai_idx(offset,tid) |
      __read_offset(result)/sizeof(rtype) == lf_bi_idx(offset,tid))
    ),
    __invariant(__implies(__write(result) & (offset < N),
      __write_offset(result)/sizeof(rtype) == lf_ai_idx(offset,tid) |
      __write_offset(result)/sizeof(rtype) == lf_bi_idx(offset,tid))
    ),
#endif
#ifdef CHECK_BI
    __invariant(upsweep_barrier(tid,/*offset=*/N,ghostsum,len)),
    __invariant(downsweep_barrier(tid,offset,result,ghostsum)),
#endif
    d < N;
    d <<= 1) {
    offset >>= 1;
#ifdef CHECK_BI_ACCESS
    upsweep_barrier_permissions(tid,/*offset=*/N,ghostsum,len)
    downsweep_barrier_permissions(tid,mul2(offset),result,ghostsum)
#endif
#ifdef CHECK_BI
    __barrier_invariant(upsweep_barrier(tid,/*offset=*/N,ghostsum,len), tid);
    __barrier_invariant(downsweep_barrier(tid,mul2(offset),result,ghostsum), tid, lf_ai_tid(tid), lf_bi_tid(tid)),
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < (d - 1)) {
      dtype ai = (offset * (t + 1)) - 1;
      dtype bi = ai + (offset >> 1);
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
  __non_temporal(__assert(__implies(tid < other_tid, raddf(result[tid], len[tid+1]) <= result[other_tid])));
#else
  #error SPEC_THREADWISE|SPEC_ELEMENTWISE must be defined
#endif
#endif

#ifdef FORCE_FAIL
  __assert(false);
#endif
}
