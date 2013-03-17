#ifndef NUM
  #error Must define NUM
#endif

#define tid get_local_id(0)
#define other_tid __other_int(tid)

__kernel void qsort(__global int *A, unsigned i, unsigned j, __global unsigned *next_pivot) {
  __local unsigned flag[NUM];
  __local unsigned scan[NUM];
  __local unsigned prescan[NUM];
  unsigned nleft;
  unsigned idx;

  int pivot = A[i];
  int val = A[tid];
  flag[tid] = val < pivot;
  bool inrange = (i <= tid) && (tid <= j);

  // hillis-steele scan (inclusive prefix sum)
  int temp;
  scan[tid] = inrange ? flag[tid] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid > 1) temp = scan[tid-1];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid > 1) scan[tid] += temp;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid > 2) temp = scan[tid-2];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid > 2) scan[tid] += temp;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid > 4) temp = scan[tid-4];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid > 4) scan[tid] += temp;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid > 8) temp = scan[tid-8];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid > 8) scan[tid] += temp;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid > 16) temp = scan[tid-16];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid >  16) scan[tid] += temp;
  barrier(CLK_LOCAL_MEM_FENCE);
  // turn into prescan (exclusive prefix sum)
  nleft = scan[NUM-1];
  prescan[tid] = tid == 0 ? 0 : scan[tid-1];

  // partition
  if (inrange) {
    if (flag[tid]) idx = i + prescan[tid];
    else           idx = i + nleft + (tid-i) - prescan[tid];
    A[idx] = val;
  }
  if (tid == i) {
    *next_pivot = idx;
  }
}
