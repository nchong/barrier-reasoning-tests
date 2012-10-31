#include "choose.h"
#define MAXWRITE 4

__kernel void compact(__global int*out, __global int*in,
  __local unsigned *num,
  __local unsigned *idx,
  unsigned n) {

  unsigned t = get_local_id(0);

  // (i) number of times to repeat element
  num[t] = CHOOSE(in[t], MAXWRITE);

  // (ii) compute indexes for scatter
  //      using an exclusive prefix sum
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < n/2) {
    idx[2*t]   = num[2*t];
    idx[2*t+1] = num[2*t+1];
  }
  // (a) upsweep
  int offset = 1;
  for (unsigned d = n/2; d > 0; d /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;
      idx[bi] += idx[ai];
    }
    offset *= 2;
  }
  // (b) downsweep
  if (t == 0) idx[n-1] = 0;
  for (unsigned d = 1; d < n; d *= 2) {
    offset /= 2;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;
      int temp = idx[ai];
      idx[ai] = idx[bi];
      idx[bi] += temp;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // end of exclusive prefix sum of flag into idx

  // (iii) repeat element num times
  for (unsigned i = 0; i < num[t]; ++i) {
    out[idx[t]+i] = in[t];
  }
}
