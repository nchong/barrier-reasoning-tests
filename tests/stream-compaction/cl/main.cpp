#include "clwrapper.h"
#ifdef __KLEE
#include <klee/klee.h>
#endif

#include <cassert>
#include <cstdio>
#include <iostream>

int main(int, char **) {
  // platform info
  std::cout << clinfo();

  // test data
  unsigned N = 4;
  size_t ArraySize = N * sizeof(int);
  int *in  = (int *)malloc(ArraySize);
  int *out = (int *)malloc(ArraySize);
#ifdef __KLEE
  klee_make_symbolic(in, ArraySize, "in");
#endif

  // initialise for device 0 on platform 0, with profiling off
  // this creates a context and command queue
  unsigned platform = 0;
  unsigned device = 0;
  bool profiling = false;
  CLWrapper clw(platform, device, profiling);

  // compile the OpenCL code
  const char *filename = "compact.cl";
  cl_program program = clw.compile(filename);

  // get kernel handle
  cl_kernel k = clw.create_kernel(program, "compact");

  // create some memory objects on the device
  cl_mem d_in   = clw.dev_malloc(ArraySize, CL_MEM_READ_ONLY);
  cl_mem d_out  = clw.dev_malloc(ArraySize);

  // memcpy into these objects
  clw.memcpy_to_dev(d_in, ArraySize, in);

  // set kernel arguments
  clw.kernel_arg(k, d_out, d_in, ArraySize, ArraySize, N);

  // run the kernel
  cl_uint dim = 1;
  size_t global_work_size = N;
  size_t local_work_size  = N;
  clw.run_kernel(k, dim, &global_work_size, &local_work_size);

  // memcpy back the result
  clw.memcpy_from_dev(d_out, ArraySize, out);

  // check results
  unsigned idx = 0;
  for (unsigned i=0; i<N; ++i) {
    if ((in[i] & 1) == 0) {
      assert(out[idx] == in[i]);
      idx++;
    }
  }
  printf("TEST PASSED\n");

  // cleanup
  free(in);
  free(out);
  // device objects will be auto-deleted when clw is destructed
  // or, we can do it manually like this:
  // clw.dev_free(d_x);
  return 0;
}

