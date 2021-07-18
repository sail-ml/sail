#include "kernels/Copy.h"
#include "Tensor.h"
#include "factories.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "loops.h"
#include "ops/broadcast.h"
#include "slice.h"
#include "tensor_shape.h"

namespace sail {

namespace internal {

namespace {

void copy_kernel(const Tensor &t1, Tensor &out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        struct Impl {
            inline void call_base(T &x1, T &out) { out = x1; }
        };

        native::UnaryElementwise<T>(Impl{}, t1, out);
    });
}

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

            T_in *p1;
            T_out *p2;

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
// std::tuple<Tensor, std::vector<std::vector<long>>>
Tensor _pad_simple(const Tensor &base, Tensor &pad_width) {
    std::vector<long> new_shape;
    long loop_size = pad_width.get_shape().shape[0];

    for (long i = 0; i < loop_size; i += 1) {
        Tensor x = pad_width[i];
        new_shape.push_back(base.get_shape()[i] + x[0].get<long>() +
                            x[1].get<long>());
    }

    Tensor padded = zeros(TensorShape(new_shape), base.get_dtype());

    std::vector<std::vector<long>> original_area;
    for (long i = 0; i < loop_size; i += 1) {
        Tensor x = pad_width[i];
        long left = x[0].get<long>();
        long size = base.get_shape()[i];
        std::vector<long> slice = {left, left + size};
        original_area.push_back(slice);
    }

    Slice s = Slice(original_area);
    padded.slice(s).assign(base);
    return padded;
}

Tensor pad_kernel(Tensor &t1, std::vector<std::vector<long>> pads) {
    std::vector<long> flat;
    for (std::vector<long> inner : pads) {
        for (long i : inner) {
            flat.push_back(i);
        }
    }

    void *pads_ptr = (void *)flat.data();
    long pad_size = static_cast<long>(pads.size());
    long pad_0_size = static_cast<long>(pads[0].size());
    Tensor pad_tensor_ =
        from_data(pads_ptr, Dtype::sInt64, TensorShape({pad_size, pad_0_size}));
    Tensor pad_tensor =
        ops::broadcast_to(pad_tensor_, TensorShape({t1.get_ndim(), 2}));

    Tensor o = _pad_simple(t1, pad_tensor);
    return o;
}

}  // namespace
REGISTER_ARCH_DISPATCH(copy_stub, DEFAULT, &copy_kernel);
// REGISTER_ONLY_NATIVE_DISPATCH(copy_stub, &copy_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(cast_stub, &cast_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(pad_stub, &pad_kernel);

}  // namespace internal

}  // namespace sail