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

class MeanSquaredErrorKernel : public Kernel {
   public:
    void execute(Tensor& t1, Tensor& t2, Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            long numel = out_tensor.numel();
            struct Impl {
                long _numel;
                Impl(long numel) : _numel(numel) {}
                inline void call_base(T x1, T x2, T& out) {
                    T ex = (T)std::pow((double)(x1 - x2), 2.0);
                    out = ex / (double)_numel;
                }
            };
            BinaryElementwiseNoAvx<T, T, T>(Impl{numel}, true, t1, t2, out_tensor);
        });
    }
};

}  // namespace sail
