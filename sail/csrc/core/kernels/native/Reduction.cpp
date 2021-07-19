// allow-no-header

#include "kernels/Reduction.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "tensor_shape.h"

namespace sail {

namespace internal {

namespace {

void sum_kernel(const Tensor& t1, std::vector<long> axis, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using avx_name = typename decltype(pt)::avx_type;
        struct Impl {
            inline void call_base(T x1, T& out) { out = out + x1; }
        };

        if (axis[0] == NULLDIM) {
            native::Reduction<T>(Impl{}, t1, out);
        } else {
            native::Reduction<T>(Impl{}, t1, out, axis);
        }
    });
}

void mean_kernel(const Tensor& t1, std::vector<long> axis, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using avx_name = typename decltype(pt)::avx_type;
        T numel = (T)t1.numel();
        struct Impl {
            T numel;
            Impl(T _numel) { numel = _numel; }
            inline void call_base(T x1, T& out) { out = out + (x1 / numel); }
        };

        if (axis[0] != NULLDIM) {
            numel = 1;
            TensorShape t1_shape = t1.get_shape();
            int i = 0;
            for (long s : t1_shape.shape) {
                if (std::find(axis.begin(), axis.end(), i) != axis.end()) {
                    numel *= s;
                }
                i += 1;
            }
            native::Reduction<T>(Impl{numel}, t1, out, axis);
        } else {
            native::Reduction<T>(Impl{numel}, t1, out);
        }
    });
}

void min_kernel(const Tensor& t1, std::vector<long> axis, Tensor& out) {
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

        if (axis[0] == NULLDIM) {
            native::Reduction<T>(Impl{}, t1, out);
        } else {
            native::Reduction<T>(Impl{}, t1, out, axis);
        }
    });
}

void max_kernel(const Tensor& t1, std::vector<long> axis, Tensor& out) {
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

        if (axis[0] == NULLDIM) {
            native::Reduction<T>(Impl{}, t1, out);
        } else {
            native::Reduction<T>(Impl{}, t1, out, axis);
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