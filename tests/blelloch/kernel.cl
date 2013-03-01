/*
 * Blelloch exclusive prefix sum in OpenCL
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
    upsweep_barrier_permissions(tid,offset,result,len)
    __barrier_invariant(upsweep_barrier(tid,offset,result,len), tid, 2*tid, 2*tid+1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < d) {
      dtype ai = offset * (2 * t + 1) - 1;
      dtype bi = offset * (2 * t + 2) - 1;
#if defined(INC_ENDSPEC) && defined(BINOP_ADD)
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
    upsweep_barrier_permissions(tid,/*offset=*/N,ghostsum,len)
    downsweep_barrier_permissions(tid,offset,result,ghostsum)
    __barrier_invariant(upsweep_barrier(tid,/*offset=*/N,ghostsum,len), tid);
    __barrier_invariant(downsweep_barrier(tid,offset,result,ghostsum), tid, div2(tid));
    barrier(CLK_LOCAL_MEM_FENCE);
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
//__assert(offset == 1);
#elif INC_ENDSPEC
  __assume(upsweep_barrier(tid,/*offset=*/N,ghostsum,len));
  __assume(downsweep_barrier(tid,/*offset=*/0,result,ghostsum));
#endif

#ifdef INC_ENDSPEC
#if defined(SPEC_THREADWISE)
  __barrier_invariant(final_upsweep_barrier(tid,ghostsum,len), upsweep_instantiation);
  __barrier_invariant(final_downsweep_barrier(tid,result,ghostsum), tid, other_tid);
  barrier(CLK_LOCAL_MEM_FENCE);
  __non_temporal(__assert(raddf(result[2*tid], len[2*tid]) == result[2*tid+1]));
  __non_temporal(__assert(__implies(tid < other_tid, raddf(result[2*tid+1], len[2*tid+1]) <= result[2*other_tid])));
#elif defined(SPEC_ELEMENTWISE)
  __barrier_invariant(final_upsweep_barrier(tid,ghostsum,len), upsweep_instantiation);
  __barrier_invariant(final_downsweep_barrier(tid,result,ghostsum), x2t(tid), x2t(other_tid));
  barrier(CLK_LOCAL_MEM_FENCE);
  __non_temporal(__assert(__implies(tid < other_tid, raddf(result[tid], len[tid]) <= result[other_tid])));
#else
  #error SPEC_THREADWISE|SPEC_ELEMENTWISE must be defined
#endif
#endif

#ifdef FORCE_FAIL
  __assert(false);
#endif
}
