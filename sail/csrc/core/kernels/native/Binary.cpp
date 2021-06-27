#include "kernels/Binary.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"

namespace sail {

namespace internal {

namespace {

void add_kernel(const Tensor& t1, const Tensor& t2, Tensor& out,
                bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, T& out) { out = x1 + x2; }
        };
        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out);
    });
}

void subtract_kernel(const Tensor& t1, const Tensor& t2, Tensor& out,
                     bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, T& out) { out = x1 - x2; }
        };
        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out);
    });
}

void multiply_kernel(const Tensor& t1, const Tensor& t2, Tensor& out,
                     bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, T& out) { out = x1 * x2; }
        };
        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out);
    });
}

void divide_kernel(const Tensor& t1, const Tensor& t2, Tensor& out,
                   bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, T& out) { out = x1 / x2; }
        };
        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out);
    });
}

}  // namespace
REGISTER_ARCH_DISPATCH(add_stub, DEFAULT, &add_kernel);
REGISTER_ARCH_DISPATCH(subtract_stub, DEFAULT, &subtract_kernel);
REGISTER_ARCH_DISPATCH(multiply_stub, DEFAULT, &multiply_kernel);
// REGISTER_ARCH_DISPATCH(divide_stub, DEFAULT, &divide_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(divide_stub, &divide_kernel);

}  // namespace internal

}  // namespace sail