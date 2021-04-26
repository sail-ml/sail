//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/std/semaphore>

#include <cuda/std/semaphore>
#include <cuda/std/chrono>

#include "test_macros.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template<typename Semaphore,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__
void test()
{
  Selector<Semaphore, Initializer> sel;
  SHARED Semaphore * s;
  s = sel.construct(0);

  auto const start = cuda::std::chrono::high_resolution_clock::now();

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  assert(!s->try_acquire_until(start + cuda::std::chrono::milliseconds(250)));
  assert(!s->try_acquire_for(cuda::std::chrono::milliseconds(250)));
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif

  auto releaser = LAMBDA (){
    //cuda::std::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s->release();
    //cuda::std::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s->release();
  };

  auto acquirer = LAMBDA (){
    assert(s->try_acquire_until(start + cuda::std::chrono::seconds(2)));
    assert(s->try_acquire_for(cuda::std::chrono::seconds(2)));
  };

  concurrent_agents_launch(acquirer, releaser);

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  auto const end = cuda::std::chrono::high_resolution_clock::now();
  assert(end - start < cuda::std::chrono::seconds(10));
#ifdef __CUDA_ARCH__
  }
#endif
}

int main(int, char**)
{
#ifndef __CUDA_ARCH__
  cuda_thread_count = 2;

  test<cuda::std::counting_semaphore<>, local_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_block>, local_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_device>, local_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_system>, local_memory_selector>();
#else
  test<cuda::std::counting_semaphore<>, shared_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_block>, shared_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_device>, shared_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_system>, shared_memory_selector>();

  test<cuda::std::counting_semaphore<>, global_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_block>, global_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_device>, global_memory_selector>();
  test<cuda::counting_semaphore<cuda::thread_scope_system>, global_memory_selector>();
#endif

  return 0;
}
