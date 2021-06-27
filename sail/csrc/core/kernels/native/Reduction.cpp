#include "kernels/Reduction.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "tensor_shape.h"

namespace sail {

namespace internal {

namespace {

void sum_kernel(const Tensor& t1, const int axis, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        // std::cout << decltype(pt)::type << std::endl;
        using T = typename decltype(pt)::type;
        using avx_name = typename decltype(pt)::avx_type;
        struct Impl {
            inline void call_base(T x1, T& out) { out = out + x1; }
        };

        if (axis != NULLDIM) {
            native::Reduction<T>(Impl{}, t1, out, axis);
        } else {
            native::Reduction<T>(Impl{}, t1, out);
        }
    });
}

void mean_kernel(const Tensor& t1, const int axis, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using avx_name = typename decltype(pt)::avx_type;
        T numel = (T)t1.numel();
        struct Impl {
            T numel;
            Impl(T _numel) { numel = _numel; }
            inline void call_base(T x1, T& out) { out = out + (x1 / numel); }
        };

        if (axis != NULLDIM) {
            int axis2 = axis;
            if (axis < 0) {
                axis2 = axis + t1.get_ndim();
            }
            numel = 1;
            TensorShape t1_shape = t1.get_shape();
            for (int i = 0; i < t1.get_ndim(); i++) {
                if (i == axis2) {
                    numel = t1_shape.shape[i];
                }
            }
            native::Reduction<T>(Impl{numel}, t1, out, axis);
        } else {
            native::Reduction<T>(Impl{numel}, t1, out);
        }
    });
}

void min_kernel(const Tensor& t1, const int axis, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using avx_name = typename decltype(pt)::avx_type;
        struct Impl {
            inline void call_base(T x1, T& out) {
                if (x1 < out) {
                    out = x1;
                }
            }
        };

        if (axis != NULLDIM) {
            native::Reduction<T>(Impl{}, t1, out, axis);
        } else {
            native::Reduction<T>(Impl{}, t1, out);
        }
    });
}

void max_kernel(const Tensor& t1, const int axis, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using avx_name = typename decltype(pt)::avx_type;
        struct Impl {
            inline void call_base(T x1, T& out) {
                if (x1 > out) {
                    out = x1;
                }
            }
        };

        if (axis != NULLDIM) {
            native::Reduction<T>(Impl{}, t1, out, axis);
        } else {
            native::Reduction<T>(Impl{}, t1, out);
        }
    });
}

}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(sum_stub, &sum_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(mean_stub, &mean_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(min_stub, &min_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(max_stub, &max_kernel);

}  // namespace internal

}  // namespace sail