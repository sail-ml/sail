#pragma once

#include <immintrin.h>
#include <cassert>  // needed for xsimd
#include <cmath>    // needed for xsimd
#include "Tensor.h"
#include "kernels/base.h"
#include "kernels/elementwise.h"
#include "kernels/unary.h"
#include "xsimd/xsimd.hpp"
#ifdef MKL
#include <mkl.h>
#include <omp.h>

#endif
namespace sail {

class PowerKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
                 bool broadcast) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                inline void call_base(T x1, T x2, T& out) {
                    out = (T)std::pow((double)x1, (double)x2);
                }
                inline void call_avx_aligned(T* x1, T* x2, T* out) {
                    out[0] = (T)std::pow((double)x1[0], (double)x2[0]);
                }
            };
            BinaryElementwiseNoAvx<T, T, T>(Impl{}, broadcast, t1, t2,
                                            out_tensor);
        });
    }
};

class PowerExpKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;
            // #ifdef MKL
            //             if (DtypeType::GetName() == "float64") {
            //                 vdExp(t1.numel(), t1.get_data(),
            //                 out_tensor.get_data()); return;
            //             } else if (DtypeType::GetName() == "float32") {
            //                 vsExp(t1.numel(), t1.get_data(),
            //                 out_tensor.get_data()); return;
            //             }
            // #endif
            struct Impl {
                inline void call_base(T x1, T& out) {
                    out = (T)std::exp((double)x1);
                }
            };
            Unary<T, T>(Impl{}, t1, out_tensor);
        });
    }
};

class LogKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;
            // #ifdef MKL
            //             if (DtypeType::GetName() == "float64") {
            //                 vdLn(t1.numel(), t1.get_data(),
            //                 out_tensor.get_data()); return;
            //             } else if (DtypeType::GetName() == "float32") {
            //                 vsLn(t1.numel(), t1.get_data(),
            //                 out_tensor.get_data()); return;
            //             }
            // #endif

            struct Impl {
                inline void call_base(T x1, T& out) {
                    out = (T)std::log((double)x1);
                }
            };
            Unary<T, T>(Impl{}, t1, out_tensor);
        });
    }
};

}  // namespace sail