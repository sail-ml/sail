#include "kernels/Conv.h"
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "ops/ops.h"

namespace sail {

namespace internal {

namespace {

Tensor im2col_kernel(Tensor &input, std::vector<long> kernel_size,
                     std::vector<long> stride, std::vector<long> pads) {
    Tensor padded_input = ops::pad(input, {pads});

    // std::vector<long> flat;
    // for (std::vector<long> inner : pads) {
    //     for (long i : inner) {
    //         flat.push_back(i);
    //     }
    // }

    // void *pads_ptr = (void *)flat.data();
    // Tensor pad_tensor_ = from_data(pads_ptr, Dtype::sInt64,
    //                                TensorShape({pads.size(),
    //                                pads[0].size()}));
    // Tensor pad_tensor =
    //     ops::broadcast_to(pad_tensor_, TensorShape({input.get_ndim(), 2}));

    // // numerator = in_dim + pad * 2 - kernel_size + stride - 1;
    // std::vector<long> out_shape;
    // for (int i = 0; i < input.get_ndim(); i++) {
    //     long numerator =
    //         input.get_shape()[i + 2] + pad_tensor[] out_shape.push_back()
    // }

    // dispatch_all_types(t1.get_dtype(), [&](auto pt) {
    //     using DtypeType = decltype(pt);
    //     using T = typename DtypeType::type;

    //     struct Impl {
    //         T min;
    //         Impl(T min) : min(min){};
    //         inline void call_base(T x1, T& out) {
    //             if (x1 < min) {
    //                 out = min;
    //                 return;
    //             }
    //             out = x1;
    //         }
    //     };
    //     native::UnaryElementwise<T>(Impl{(T)min}, t1, out);
    // });
    return empty_like(input);
}

}  // namespace

REGISTER_ONLY_NATIVE_DISPATCH(im2col_stub, &im2col_kernel);

}  // namespace internal

}  // namespace sail