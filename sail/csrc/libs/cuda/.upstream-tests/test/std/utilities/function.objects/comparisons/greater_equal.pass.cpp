//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// greater_equal

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#ifndef __CUDACC_RTC__
#include "pointer_comparison_test_helper.hpp"
#endif

int main(int, char**)
{
    typedef cuda::std::greater_equal<int> F;
    const F f = F();
    static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((cuda::std::is_same<bool, F::result_type>::value), "" );
    assert(f(36, 36));
    assert(f(36, 6));
    assert(!f(6, 36));
#ifndef __CUDA_ARCH__
    {
        // test total ordering of int* for greater_equal<int*> and
        // greater_equal<void>.
        do_pointer_comparison_test<int, cuda::std::greater_equal>();
    }
#endif
#if TEST_STD_VER > 11
    typedef cuda::std::greater_equal<> F2;
    const F2 f2 = F2();
    assert(f2(36, 36));
    assert(f2(36, 6));
    assert(!f2(6, 36));
    assert( f2(36, 6.0));
    assert( f2(36.0, 6));
    assert(!f2(6, 36.0));
    assert(!f2(6.0, 36));

    constexpr bool foo = cuda::std::greater_equal<int> () (36, 36);
    static_assert ( foo, "" );

    constexpr bool bar = cuda::std::greater_equal<> () (36.0, 36);
    static_assert ( bar, "" );
#endif

  return 0;
}
