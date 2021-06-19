#pragma once

#include <immintrin.h>
#include <cassert>  // needed for xsimd
#include "Tensor.h"
#include "kernels/base.h"
#include "kernels/unary.h"
#include "xsimd/xsimd.hpp"
namespace sail {

class ClipMinOnlyKernel : public Kernel {
   public:
    void execute(const Tensor& t1, double value, const Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                T min_val;
                Impl(T min_val_) : min_val(min_val_) {}
                inline void call_base(T x1, T& out) {
                    if (x1 >= min_val) {
                        out = x1;
                    } else {
                        out = min_val;
                    }
                }
            };
            Unary<T, T>(Impl{(T)value}, t1, out_tensor);
        });
    }
};

class ClipKernel : public Kernel {
   public:
    void execute(const Tensor& t1, double min_val, double max_val,
                 const Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                T max_val, min_val;
                Impl(T min_val_, T max_val_)
                    : min_val(min_val_), max_val(max_val_) {}
                inline void call_base(T x1, T& out) {
                    if ((min_val <= x1) && (x1 <= max_val)) {
                        out = x1;
                    } else {
                        if (x1 < min_val) {
                            out = min_val;
                        } else {
                            out = max_val;
                        }
                    }
                }
            };
            Unary<T, T>(Impl{(T)min_val, (T)max_val}, t1, out_tensor);
        });
    }
};

}  // namespace sail
