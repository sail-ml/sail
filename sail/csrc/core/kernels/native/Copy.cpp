#include "kernels/Copy.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "tensor_shape.h"

namespace sail {

namespace internal {

namespace {

void cast_kernel(const Tensor &t1, Tensor &out_tensor) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        dispatch_all_types(out_tensor.get_dtype(), [&](auto xt) {
            using T_in = typename decltype(pt)::type;
            using T_out = typename decltype(xt)::type;
            int numel = t1.get_shape().numel();
            int i;

            struct Impl {
                inline void call_base(T_in &x1, T_out &out) {
                    out = static_cast<T_out>(x1);
                }
            };

            T_in __restrict__ *p1;
            T_out __restrict__ *p2;

            p1 = static_cast<T_in *>(t1.get_data());
            p2 = static_cast<T_out *>(out_tensor.get_data());

            Impl op = Impl{};

            if (t1.is_view()) {
                TensorShape s = t1.get_shape();
                MultiTensorIterator iter = MultiTensorIterator(s);
                int inner_loop_size = iter.inner_loop_size();
                int outer_steps = iter.out_loop_size();

                int z = 0;
                for (int i = 0; i < outer_steps; i++) {
                    for (int j = 0; j < inner_loop_size; j += 1) {
                        op.call_base(p1[iter.d_ptrs[0]], p2[z]);
                        iter.advance_d_ptr(1);
                        z += 1;
                    }
                    iter.backup_d_ptr();
                    iter.next();
                }
            } else {
                for (i = 0; i < numel; i += 1) {
                    op.call_base(p1[i], p2[i]);
                }
            }
        });
    });
}

}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(cast_stub, &cast_kernel);

}  // namespace internal

}  // namespace sail