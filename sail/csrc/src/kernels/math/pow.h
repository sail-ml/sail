#pragma once

#include <immintrin.h>
#include <cassert>  // needed for xsimd
#include <cmath>  // needed for xsimd
#include "../../Tensor.h"
#include "../base.h"
#include "../unary.h"
#include "xsimd/xsimd.hpp"
namespace sail {

class PowerKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const double power, const Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                inline void call_base(T x1, T& out) { 
                    out = (T)std::pow((double)x1, (double)power); 
                }
            };
            Unary<T, T>(Impl{}, t1, out_tensor);
        });
    }
};

class PowerExpKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                inline void call_base(T x1, T& out) { 
                    out = (T)std::exp((double)x1); 
                }
            };
            Unary<T, T>(Impl{}, t1, out_tensor);
        });
    }
};

}  // namespace sail