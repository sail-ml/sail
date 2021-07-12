#pragma once
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/Conv.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {

inline Tensor im2col(Tensor& im2col_input, Tensor& kernel,
                     std::vector<long> stride, long pad_x, long pad_y, long b,
                     long new_height, long new_width) {
    long d = 1;

    long k_cin = kernel.get_shape()[0];
    long k_cout = kernel.get_shape()[1];
    long k_h = kernel.get_shape()[2];
    long k_w = kernel.get_shape()[3];

    std::vector<std::vector<long>> slices = {
        {},
        {},
        {0, im2col_input.get_shape()[2], stride[0]},
        {0, im2col_input.get_shape()[3], stride[1]}};

    std::vector<long> strides_ = {(long)1, 1, stride[0], stride[1]};
    auto strides_tensor = from_data((void*)strides_.data(), Dtype::sInt64,
                                    TensorShape({strides_.size()}));

    auto index_strides_ = im2col_input.slice(Slice(slices)).get_shape().strides;

    auto window_shape =
        from_data((void*)kernel.get_shape().shape.data(), Dtype::sInt64,
                  TensorShape({kernel.get_shape().shape.size()}));
    auto window_strides =
        from_data((void*)im2col_input.get_shape().strides.data(), Dtype::sInt64,
                  TensorShape({im2col_input.get_shape().shape.size()}));
    auto indexing_strides =
        from_data((void*)index_strides_.data(), Dtype::sInt64,
                  TensorShape({index_strides_.size()}));
    auto in_shape =
        from_data((void*)im2col_input.get_shape().shape.data(), Dtype::sInt64,
                  TensorShape({im2col_input.get_shape().shape.size()}));

    auto win_indices_shape = ((in_shape - window_shape) / strides_tensor) + 1;
    auto c_win_indices_shape = win_indices_shape.cast(Dtype::sInt64);
    // sail::ops::cast(win_indices_shape, Dtype::sInt64);

    auto new_shape = sail::ops::cat({c_win_indices_shape, window_shape});
    auto new_strides = sail::ops::cat({indexing_strides, window_strides});

    std::vector<long> ns((long*)new_shape.get_data(),
                         (long*)new_shape.get_data() + new_shape.numel());
    std::vector<long> ns_((long*)new_strides.get_data(),
                          (long*)new_strides.get_data() + new_strides.numel());

    auto cols2 = as_strided(im2col_input, sail::TensorShape(ns, ns_));

    auto cols = sail::ops::reshape(
        cols2,
        sail::TensorShape({new_height * new_width * b, k_cin * k_w * k_h}));
}

}  // namespace sail