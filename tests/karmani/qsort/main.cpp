#include "clwrapper.h"
#include <cassert>
#include <cstdio>
#include <iostream>

int main(int, char **) {
  // platform info
  std::cout << clinfo();

  // test data
  unsigned N = 8;
  size_t ArraySize = N * sizeof(int);
  int *A = (int *)malloc(ArraySize);
  A[0] = 3;
  A[1] = 1;
  A[2] = 7;
  A[3] = 0;
  A[4] = 4;
  A[5] = 1;
  A[6] = 6;
  A[7] = 3;
  unsigned int i = 0;
  unsigned int j = N-1;
  int next_pivot = -1;

  // initialise for device 0 on platform 0, with profiling off
  // this creates a context and command queue
  unsigned platform = 0;
  unsigned device = 0;
  bool profiling = false;
  CLWrapper clw(platform, device, profiling);

  // compile the OpenCL code
  const char *filename = "kernel.cl";
  cl_program program = clw.compile(filename);

  // get kernel handle
  cl_kernel k = clw.create_kernel(program, "qsort");

  // create some memory objects on the device
  cl_mem d_A           = clw.dev_malloc(ArraySize);
  cl_mem d_i           = clw.dev_malloc(sizeof(unsigned int));
  cl_mem d_j           = clw.dev_malloc(sizeof(unsigned int));
  cl_mem d_next_pivot  = clw.dev_malloc(sizeof(int));

  // memcpy into these objects
  clw.memcpy_to_dev(d_A, ArraySize, A);
  clw.memcpy_to_dev(d_i, sizeof(unsigned int), &i);
  clw.memcpy_to_dev(d_j, sizeof(unsigned int), &j);
  clw.memcpy_to_dev(d_next_pivot, sizeof(int), &next_pivot);

  // set kernel arguments
  clw.kernel_arg(k, d_A, d_i, d_j, d_next_pivot);

  // run the kernel
  cl_uint dim = 1;
  size_t global_work_size = N;
  size_t local_work_size  = N;
  clw.run_kernel(k, dim, &global_work_size, &local_work_size);

  // memcpy back the result
  clw.memcpy_from_dev(d_A, ArraySize, A);
  clw.memcpy_from_dev(d_next_pivot, sizeof(int), &next_pivot);

  // check results
  for (unsigned i=0; i<N; ++i) {
    printf("A[%d] = %d", i, A[i]);
  }
  printf("next_pivot = %d", next_pivot);
  printf("TEST PASSED\n");

  // cleanup
  free(A);
  // device objects will be auto-deleted when clw is destructed
  // or, we can do it manually like this:
  // clw.dev_free(d_x);
  return 0;
}

