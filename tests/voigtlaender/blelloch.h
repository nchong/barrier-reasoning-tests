template<class O>
__device__ void scan(char *input, char *output, O op) {
  __shared__ char result[N];
  __shared__ char total;

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

  if (t == 0) {
    total = result[N-1];
    result[N-1] = 0;
  }

  for (unsigned d = 1; d < N; d *= 2) {
    offset /= 2;
    __syncthreads();
    if (t < d) {
      unsigned ai = offset * (2 * t + 1) - 1;
      unsigned bi = offset * (2 * t + 2) - 1;
      char temp = result[ai];
      result[ai] = result[bi];
      result[bi] = op(result[bi],temp);
    }
  }
  __syncthreads();

  // change into inclusive prefix sum
  if (t < N/2) {
    output[2*t]   = result[2*t+1];
    output[2*t+1] = (2*t+1 == N-1) ? total: result[2*t+2];
  }
}
