template<class O>
__device__ void scan(char *input, char *output, O op) {
  __shared__ char result[N];

  unsigned t = threadIdx.x;

  result[t] = input[t];

  __syncthreads();

  char temp;
  for (unsigned offset = 1; offset < N; offset *= 2) {
    if (t >= offset) {
      temp = result[t-offset];
    }
    __syncthreads();
    if (t >= offset) {
      result[t] = op(temp, result[t]);
    }
    __syncthreads();
  }

  output[t] = result[t];
}
