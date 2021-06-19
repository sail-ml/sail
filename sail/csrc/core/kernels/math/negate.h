#pragma once

#include <immintrin.h>
#include <cassert>  // needed for xsimd
#include "Tensor.h"
#include "kernels/base.h"
#include "kernels/unary.h"
#include "xsimd/xsimd.hpp"
namespace sail {

template <typename T>
using dispatch = void (*)(T, T&);

template <typename T>
void negate(T x, T& out) {
    out = -x;
}

template <typename T>
dispatch<T> dispatched_negate = negate<T>;

class NegateTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                // operator()(T x1, T& out) {
                //     return (*dispatched_negate)(x1, out);
                // }
                void call_base(T x1, T& out) {
                    return (*dispatched_negate<T>)(x1, out);
                }
            };
            Unary<T, T>(Impl{}, t1, out_tensor);
        });
    }
};

}  // namespace sail
