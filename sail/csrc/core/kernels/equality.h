#pragma once

#include <immintrin.h>
#include <cassert>  // needed for xsimd
#include "../Tensor.h"
#include "base.h"
#include "elementwise.h"
#include "xsimd/xsimd.hpp"
namespace sail {

class ElementwiseEquality : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
                 bool broadcast) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                inline void call_base(T& x1, T& x2, T& out) {
                    if (x1 == x2) {
                        out = (T)1;
                    } else {
                        out = (T)0;
                    }
                }
            };

            BinaryElementwiseNoAvx<T, T, T>(Impl{}, broadcast, t1, t2,
                                            out_tensor);
        });
    }
};
}  // namespace sail