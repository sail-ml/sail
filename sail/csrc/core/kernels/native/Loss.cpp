#include "kernels/Loss.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "tensor_shape.h"

namespace sail {

namespace internal {

namespace {

void mse_kernel(const Tensor& t1, const Tensor& t2, Tensor& out_tensor) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        bool broadcast = false;
        if (t1.is_view() || t2.is_view()) {
            broadcast = true;
        }

        long numel = out_tensor.numel();
        struct Impl {
            long _numel;
            Impl(long numel) : _numel(numel) {}
            inline void call_base(T x1, T x2, T& out) {
                T ex = (T)std::pow((double)(x1 - x2), 2.0);
                out = ex / (double)_numel;
            }
        };
        native::BinaryElementwise<T>(Impl{numel}, broadcast, t1, t2,
                                     out_tensor);
    });
}

}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(mse_stub, &mse_kernel);

}  // namespace internal

}  // namespace sail