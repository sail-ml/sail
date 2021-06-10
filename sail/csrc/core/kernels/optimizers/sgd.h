#pragma once

#include <immintrin.h>
#include <cassert>  // needed for xsimd
#include <cmath>    // needed for xsimd
#include "../../Tensor.h"
#include "../base.h"
#include "../elementwise.h"
#include "../unary.h"
#include "xsimd/xsimd.hpp"
#ifdef MKL
#include <mkl.h>
#include <omp.h>

#endif
namespace sail {

template <typename T, typename Op>
void launch_optimizer_broadcast(Op op, const Tensor& t1, const Tensor& grad) {
    // why this works:
    // PARAMETERS ARE NEVER BROADCASTED VALUES
    int i = 0;

    T* p1;
    T* p_grad;

    p1 = static_cast<T*>(t1.get_data());
    p_grad = static_cast<T*>(grad.get_data());

    TensorShape s1 = TensorShape(t1.get_shape());
    TensorShape s2 = TensorShape(grad.get_shape());
    int numel = t1.get_shape().numel();

    s2.recompute();

    // T* data = (T*)_malloc_align(numel, t1.get_info().alignment,
    // t1.get_info().dtype_size);

    for (i = 0; i < numel; i += 1) {
        op.call_base(p1[i], p_grad[s2.d_ptr]);
        // s1.next();
        s2.next();
    }
}

class SGDKernel : public Kernel {
   public:
    void execute(Tensor& t1, Tensor& grad, const float learning_rate) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            struct Impl {
                T lr;
                Impl(T _lr) : lr(_lr) {}
                inline void call_base(T& x1, T& grad) {
                    x1 = x1 + (-lr * grad);
                }
            };
            launch_optimizer_broadcast<T>(Impl{(T)learning_rate}, t1, grad);
        });
    }
};

}  // namespace sail