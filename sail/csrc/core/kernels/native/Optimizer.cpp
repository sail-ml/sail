// allow-no-header

#include "kernels/Optimizer.h"
#include "Tensor.h"
#include "dtypes.h"
#include "exception.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"

namespace sail {

namespace internal {

namespace {

void sgd_update_kernel(Tensor& t1, Tensor& grad, const float learning_rate) {
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        SAIL_CHECK_LINE(t1.is_view() == false, "parameter must not be a view");
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            T lr;
            Impl(T _lr) : lr(_lr) {}
            inline void call_base(T& x1, T& grad) { x1 = x1 + (-lr * grad); }
        };
        native::BinaryElementwiseInPlace<T>(Impl{(T)learning_rate}, t1, grad);
    });
}
}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(sgd_stub, &sgd_update_kernel);

}  // namespace internal

}  // namespace sail