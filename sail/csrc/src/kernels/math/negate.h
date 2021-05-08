#pragma once

#include <immintrin.h>
#include <cassert>  // needed for xsimd
#include "../../Tensor.h"
#include "../base.h"
#include "../unary.h"
#include "xsimd/xsimd.hpp"
namespace sail {

class NegateTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& out_tensor) {
        launch_arithmetic(t1.dtype, [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                inline void call_base(T x1, T& out) { out = -x1; }
            };
            Unary<T, T>(Impl{}, t1, out_tensor);
        });
    }
};

}  // namespace sail
