/*
 * Kogge Stone inclusive prefix sum in OpenCL
 */

#define N __LOCAL_SIZE_0
#define tid get_local_id(0)
#define other_tid __other_int(tid)

#define bitrange(a,b) \
  (((1 << (b-a+1)) - 1) << a)

#define sweep(t,offset) \
  __ite(t < offset, ghostsum[t] == bitrange(0,t), \
                    ghostsum[t] == bitrange((t-offset+1),t))

#define isthreadid(t) (0 <= t & t < N)

__kernel void scan(__global int *input, __global int *output) {
  __local int sum[N];
  __local long int ghostsum[N];

  sum[tid] = input[tid];
  ghostsum[tid] = 1<<tid;

  __assert(__accessed(input, tid));
  __barrier_invariant(sweep(tid,1), tid, tid-1);
  barrier(CLK_LOCAL_MEM_FENCE);

  int temp;
  int ghosttemp;

  for (int offset = 1;
        __invariant(__no_read(output)), __invariant(__no_write(output)),
        __invariant(__no_read(sum)), __invariant(__no_write(sum)),
        __invariant(__no_read(ghostsum)), __invariant(__no_write(ghostsum)),
        __invariant(0 <= offset),
        __invariant(__is_pow2(offset)),
        __invariant(offset <= N),
        __invariant(sweep(tid,offset)),
        __invariant(__implies(isthreadid(tid-offset), sweep(tid-offset,offset))),
      offset < N;
      offset *= 2) 
  {
    if (tid >= offset)
    {
      temp = sum[tid-offset];
      ghosttemp = ghostsum[tid-offset];
    }

    __read_permission(ghostsum[tid]);
    __barrier_invariant(sweep(tid,offset), tid, tid-offset, other_tid);
    __barrier_invariant(__implies(tid >= offset, temp == sum[tid-offset]), tid);
    __barrier_invariant(__implies(tid >= offset, ghosttemp == ghostsum[tid-offset]), tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid >= offset)
    {
      // concretely
      sum[tid] = __add_noovfl_int(sum[tid], temp);
      // abstractly, adding adjacent intervals
      ghostsum[tid] |= ghosttemp;
    }

    __read_permission(ghostsum[tid]);
    __barrier_invariant(sweep(tid,2*offset), tid, tid-(2*offset));
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  output[tid] = sum[tid];
  __assert(__accessed(output, tid));

#ifdef FORCE_FAIL
  __assert(false);
#endif
}
