#include "kernels/Unary.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"

namespace sail {

namespace internal {

namespace {

void negate_kernel(const Tensor& t1, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T x1, T& out) { out = -x1; }
        };
        native::UnaryElementwise<T>(Impl{}, t1, out);
    });
}

}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(negate_stub, &negate_kernel);

}  // namespace internal

}  // namespace sail