template<class O>
__device__ void scan(char *input, char *output, O op) {
  __shared__ char result[N];
  unsigned offset;
  unsigned t = threadIdx.x;

  if (t < N/2) {
    result[2*t]   = input[2*t];
    result[2*t+1] = input[2*t+1];
  }

  offset = 1;
  for (unsigned d = N/2; d > 0; d /= 2) {
    __syncthreads();
    if (t < d) {
      unsigned ai = offset * (2 * t + 1) - 1;
      unsigned bi = offset * (2 * t + 2) - 1;
      result[bi] = op(result[ai],result[bi]);
    }
    offset *= 2;
  }

  for (unsigned d = 2; d < N; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (t < (d - 1)) {
      unsigned ai = (offset * (t + 1)) - 1;
      unsigned bi = ai + (offset >> 1);
      result[bi] = op(result[ai], result[bi]);
    }
  }
  __syncthreads();

  if (t < N/2) {
    output[2*t]   = result[2*t];
    output[2*t+1] = result[2*t+1];
  }
}
