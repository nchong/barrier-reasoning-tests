#ifndef N
#define N 4
#endif

#define __1D_WORK_GROUP
#define __1D_GRID
#include "opencl.h"
__axiom(get_local_size(0) == N);
__axiom(get_num_groups(0) == 1);

#define tid get_local_id(0)
#define other_tid __other_int(tid)

__kernel void qsort(__local int *A, __local int *B) {
  __local unsigned int flag[N];
  __local unsigned int scan[N];
  __local unsigned int prescan[N];
  __local unsigned int idx[N];
  __local unsigned int nleft;

  int pivot = A[0];
  int val = A[tid];
  flag[tid] = val < pivot;
  barrier(CLK_LOCAL_MEM_FENCE);
  /*A*/ __assume(flag[tid] == 0 | flag[tid] == 1);
  /*B*/ __assume(__implies(tid < other_tid, (idx[tid] + flag[tid]) <= idx[other_tid]));
  /*C*/ __assume(__implies(tid < other_tid, __add_noovfl(idx[tid], flag[tid])));
  /*D*/ __assume(__implies(tid < other_tid, (idx[other_tid] < (idx[tid] + (other_tid - tid)))));
  /*E*/ __assume(__ite(flag[tid] == 1, idx[tid] < nleft, nleft <= (nleft + tid - idx[tid])));

  if (flag[tid]) A[idx[tid]] = val;
  else A[nleft+tid-idx[tid]] = val;
#ifdef FORCE_FAIL
  __assert(false);
#endif
}
