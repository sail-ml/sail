// allow-no-header

#include "Tensor.h"
#include "dtypes.h"
#include "exception.h"
#include "kernels/Binary.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"

namespace sail {

namespace internal {

namespace {

void power_kernel(const Tensor& t1, const Tensor& t2, Tensor& out,
                  bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T x1, T x2, T& out) {
                out = (T)std::pow((double)x1, (double)x2);
            }
        };
        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out);
    });
}

void exp_kernel(const Tensor& t1, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;
        struct Impl {
            inline void call_base(T x1, T& out) {
                out = (T)std::exp((double)x1);
            }
        };
        native::UnaryElementwise<T>(Impl{}, t1, out);
    });
}

void log_kernel(const Tensor& t1, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;
        struct Impl {
            inline void call_base(T x1, T& out) {
                out = (T)std::log((double)x1);
            }
        };
        native::UnaryElementwise<T>(Impl{}, t1, out);
    });
}

}  // namespace
REGISTER_ARCH_DISPATCH(log_stub, DEFAULT, &log_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(power_stub, &power_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(exp_stub, &exp_kernel);

}  // namespace internal

}  // namespace sail