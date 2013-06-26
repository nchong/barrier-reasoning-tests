template<class O>
__device__ void scan(char *input, char *output, O op) {
  __shared__ char result[N];

  unsigned t = threadIdx.x;

  result[2*t] = input[2*t];
  result[2*t+1] = input[2*t+1];

  for (unsigned d = 1; d < N; d *= 2) {
    __syncthreads();
    unsigned block = 2 * (t - (t & (d - 1)));
    unsigned me    = block + (t & (d - 1)) + d;
    unsigned spine = block + d - 1;
    result[me] = op(result[spine], result[me]);
  }

  output[2*t] = result[2*t];
  output[2*t+1] = result[2*t+1];
}
