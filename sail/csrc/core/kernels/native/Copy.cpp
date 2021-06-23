#include "kernels/Copy.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "tensor_shape.h"

namespace sail {

namespace internal {

namespace {

void cast_kernel(const Tensor& t1, Tensor& out_tensor) {
    launch_arithmetic(t1.get_dtype(), [&](auto pt) {
        launch_arithmetic(out_tensor.get_dtype(), [&](auto xt) {
            using T_in = typename decltype(pt)::type;
            using T_out = typename decltype(xt)::type;

            struct Impl {
                inline void call_base(T_in x1, T_in& out) {
                    out = static_cast<T_out>(x1);
                }
            };

            native::UnaryElementwise<T_in>(Impl{}, t1, out_tensor);
        });
    });
}

}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(cast_stub, &cast_kernel);

}  // namespace internal

}  // namespace sail