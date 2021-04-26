//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// multiplies

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda::std::multiplies<int> F;
    const F f = F();
    static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((cuda::std::is_same<int, F::result_type>::value), "" );
    assert(f(3, 2) == 6);
#if TEST_STD_VER > 11
    typedef cuda::std::multiplies<> F2;
    const F2 f2 = F2();
    assert(f2(3,2) == 6);
    assert(f2(3.0, 2) == 6);
    assert(f2(3, 2.5) == 7.5); // exact in binary

    constexpr int foo = cuda::std::multiplies<int> () (3, 2);
    static_assert ( foo == 6, "" );

    constexpr double bar = cuda::std::multiplies<> () (3.0, 2);
    static_assert ( bar == 6.0, "" );
#endif

  return 0;
}
