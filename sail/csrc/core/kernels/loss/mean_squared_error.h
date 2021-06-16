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

class MeanSquaredErrorBackwardKernel : public Kernel {
   public:
    void execute(Tensor& inputs1, Tensor& inputs2, Tensor& grad, Tensor& out_tensor_pos, Tensor& out_tensor_neg) {
        launch_arithmetic(inputs1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            long numel = inputs1.numel();
            struct Impl {
                long _numel;
                T v;
                Impl(long numel) : _numel(numel) {
                    v = 2.0 / (float)numel;
                }
                inline void call_base(T in1, T in2, T g, T& op, T& on) {
                    T base = (in1 - in2) * v * g;
                    op = base;
                    on = -base;
                }
            };

            Impl op = Impl{numel};

            int i = 0;

            T *in1;
            T *in2;
            T *g;
            T *o1;
            T *o2;

            in1 = static_cast<T*>(inputs1.get_data());
            in2 = static_cast<T*>(inputs2.get_data());
            g = static_cast<T*>(grad.get_data());
            o1 = static_cast<T*>(out_tensor_pos.get_data());
            o2 = static_cast<T*>(out_tensor_neg.get_data());


            TensorShape main = inputs1.get_shape();
            TensorShape grad_s = grad.get_shape();

            main.recompute();
            grad_s.recompute();

            for (i = 0; i < numel; i += 1) {
                op.call_base(
                    in1[main.d_ptr], in2[main.d_ptr], 
                    g[grad_s.d_ptr],
                    o1[main.d_ptr], o2[main.d_ptr]
                );
                main.next();
                grad_s.next();
            }

            main.reset();
            grad_s.reset();
        });
    }
};

}  // namespace sail
