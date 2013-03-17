#include "clwrapper.h"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <random>
#include <sstream>
#include <stack>
#include <utility>

#define N 8   //< number of elements to sort
#define TRACE //< turn on printing of intermediate results

void print_state(int *A, unsigned left, unsigned right, int next_pivot=-1);
void print_state(int *A, unsigned left, unsigned right, int next_pivot) {
  printf("A = [");
  for (unsigned i=0; i<N; ++i) {
    printf("%s%d%s%s",
      i == left ? "|" : "",
      A[i], i == right ? "|" : "",
      i == (N-1) ? "" : ", ");
  }
  printf("]\n");
  if (next_pivot != -1) {
    printf("next_pivot %d\n", next_pivot);
  }
}

int main(int, char **) {
  // platform info
  std::cout << clinfo() << std::endl;

  // test data
  size_t ArraySize = N * sizeof(int);
  int *A = (int *)malloc(ArraySize);
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,100);
  for (unsigned i=0; i<N; ++i) {
    A[i] = distribution(generator);
  }
  std::cout << "INITIAL STATE" << std::endl;
  print_state(A,0,N-1);

  // initialise for device 0 on platform 0, with profiling off
  // this creates a context and command queue
  unsigned platform = 0;
  unsigned device = 0;
  bool profiling = false;
  CLWrapper clw(platform, device, profiling);

  // compile the OpenCL code
  const char *filename = "kernel.cl";
  std::stringstream ss;
  ss << "-DNUM=" << N;
  cl_program program = clw.compile(filename,ss.str().c_str());

  // get kernel handle
  cl_kernel k = clw.create_kernel(program, "qsort");

  // create some memory objects on the device
  cl_mem d_A           = clw.dev_malloc(ArraySize);
  cl_mem d_next_pivot  = clw.dev_malloc(sizeof(int));

  // memcpy input
  clw.memcpy_to_dev(d_A, ArraySize, A);

  // kernel run parameters
  cl_uint dim = 1;
  size_t global_work_size = N;
  size_t local_work_size  = N;

  // worklist
  std::stack<std::pair<unsigned,unsigned> > stack;
  stack.push(std::make_pair(0,N-1));
  while (!stack.empty()) {
    std::pair<unsigned,unsigned> bounds = stack.top();
    stack.pop();
    unsigned left  = bounds.first;
    unsigned right = bounds.second;
#ifdef TRACE
    printf("Running [%d,%d]\n", left, right);
#endif
    assert(left <= right);
    if (left == right) continue;

    // set kernel arguments
    int next_pivot = -1;
    clw.kernel_arg(k, d_A, left, right, d_next_pivot);

    // run the kernel
    clw.run_kernel(k, dim, &global_work_size, &local_work_size);

    // memcpy back the result
    clw.memcpy_from_dev(d_next_pivot, sizeof(int), &next_pivot);
#ifdef TRACE
    clw.memcpy_from_dev(d_A, ArraySize, A);
    print_state(A,left,right,next_pivot);
#endif

    assert(next_pivot != -1);
    if ((int)left < (next_pivot-1)) {
#ifdef TRACE
      printf("Pushing [%d,%d]\n", left, next_pivot-1);
#endif
      stack.push(std::make_pair(left, next_pivot-1));
    }
    if (next_pivot+1 < (int)right) {
#ifdef TRACE
      printf("Pushing [%d,%d]\n", next_pivot+1, right);
#endif
      stack.push(std::make_pair(next_pivot+1, right));
    }
  }

  // check results
#ifndef TRACE
  clw.memcpy_from_dev(d_A, ArraySize, A);
#endif
  printf("END STATE\n");
  print_state(A,0,N-1);
  for (unsigned i=1; i<N; ++i) {
    assert(A[i-1] <= A[i]);
  }
  printf("TEST PASSED\n");

  // cleanup
  free(A);
  // device objects will be auto-deleted when clw is destructed
  // or, we can do it manually like this:
  // clw.dev_free(d_A);
  return 0;
}

