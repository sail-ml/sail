// allow-no-header

#include "kernels/Activation.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "ops/ops.h"

namespace sail {

namespace internal {

namespace {

void sigmoid_kernel(const Tensor& t1, Tensor& out) {
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            T one = (T)1;
            inline void call_base(T x1, T& out) {
                T neg = -x1;
                T exp_ = (T)std::exp((double)neg);
                T denom = one + exp_;
                out = one / denom;
            }
        };
        native::UnaryElementwise<T>(Impl{}, t1, out);
    });
}

void sigmoid_backward_kernel(const Tensor& t1, Tensor& out) {
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            T one = (T)1;
            inline void call_base(T x1, T& out) { out = x1 * (one - x1); }
        };
        native::UnaryElementwise<T>(Impl{}, t1, out);
    });
}

void softmax_kernel(Tensor& t1, const int axis, Tensor& out) {
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        Tensor max = ops::max(t1, axis, true);
        Tensor t2 = t1 - max;
        Tensor s = ops::sum(ops::exp(t2), axis, true);
        Tensor new_s = ops::broadcast_to(s, t2.get_shape());
        struct Impl {
            T one = (T)1;
            inline void call_base(T x1, T s_val, T& out) {
                T ex = (T)std::exp((double)x1);
                out = ex / s_val;
            }
        };
        native::BinaryElementwise<T>(Impl{}, true, t2, new_s, out);
    });
}

void softmax_backward_partial_kernel(Tensor& y, Tensor& targets,
                                     Tensor& out_tensor) {
    dispatch_all_numeric_types(y.get_dtype(), [&](auto pt) {
        dispatch_all_numeric_types(targets.get_dtype(), [&](auto pt2) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;
            using T2 = typename decltype(pt2)::type;

            int i = 0;

            long n = targets.get_shape()[0];

            T* p1;
            T2* targ;
            T* p3;

            p1 = static_cast<T*>(y.get_data());
            targ = static_cast<T2*>(targets.get_data());
            p3 = static_cast<T*>(out_tensor.get_data());

            int size = y.get_shape().shape[1];
            int numel = y.numel();

            int start = (int)targ[0];
            int prev = start;
            int j = 1;
            for (int i = 0; i < numel; i++) {
                if (i == start) {
                    p3[i] = (p1[i] - 1) / n;
                    start += size - prev;
                    start += int(targ[j]);
                    prev = int(targ[j]);
                    j += 1;
                } else {
                    p3[i] = p1[i] / n;
                }
            }
        });
    });
}
void softmax_mul_sum_kernel(Tensor& t1, Tensor& targets, Tensor& out_tensor) {
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        dispatch_all_numeric_types(targets.get_dtype(), [&](auto pt2) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;
            using T2 = typename decltype(pt2)::type;

            int i = 0;

            T* p1;
            T2* targ;
            T* p3;

            p1 = static_cast<T*>(t1.get_data());
            targ = static_cast<T2*>(targets.get_data());
            p3 = static_cast<T*>(out_tensor.get_data());

            int numel = targets.numel();

            int start = 0;
            int size = t1.get_shape().shape[1];
            for (int i = 0; i < numel; i++) {
                int jump = (int)targ[i];
                p3[0] += p1[start + jump];
                start += size;
            }
        });
    });
}
}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(softmax_stub, &softmax_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(sigmoid_stub, &sigmoid_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(softmax_backward_partial_stub,
                              &softmax_backward_partial_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(softmax_mul_sum_stub, &softmax_mul_sum_kernel);

}  // namespace internal

}  // namespace sail