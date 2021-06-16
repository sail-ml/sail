#pragma once

#include <immintrin.h>
#include <omp.h>
#include <cassert>  // needed for xsimd
#include <cmath>    // needed for xsimd
#include "Tensor.h"
#include "kernels/base.h"
#include "kernels/elementwise.h"
#include "kernels/unary.h"
#include "ops/ops.h"
#include "xsimd/xsimd.hpp"
namespace sail {

class ReluBackwardsKernel : public Kernel {
   public:
    void execute(Tensor& t1, Tensor& grad, Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                T one = (T)1;
                inline void call_base(T x1, T g, T& out) {
                    if (x1 < 0) {
                        out = 0;
                    } else {
                        out = g * x1;
                    }
                }
            };
            BinaryElementwiseNoAvx<T, T, T>(Impl{}, true, t1, grad, out_tensor);
        });
    }
};
}