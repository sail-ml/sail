#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/Conv.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {

inline Tensor im2col(Tensor& im2col_input, TensorShape kernel,
                     std::vector<long> stride, long pad_x, long pad_y, long b,
                     long new_height, long new_width) {
    long d = 1;

    long k_cin = kernel[0];
    long k_cout = kernel[1];
    long k_h = kernel[2];
    long k_w = kernel[3];

    std::vector<std::vector<long>> slices = {
        {},
        {},
        {0, im2col_input.get_shape()[2], stride[0]},
        {0, im2col_input.get_shape()[3], stride[1]}};

    std::vector<long> strides_ = {(long)1, 1, stride[0], stride[1]};
    auto strides_tensor = from_data((void*)strides_.data(), Dtype::sInt64,
                                    TensorShape({strides_.size()}));

    auto index_strides_ = im2col_input.slice(Slice(slices)).get_shape().strides;

    auto window_shape = from_data((void*)kernel.shape.data(), Dtype::sInt64,
                                  TensorShape({kernel.shape.size()}));
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

    return cols2;
}

inline Tensor col2im(Tensor& cols, TensorShape kernel,
                     std::vector<long> strides, long pad_x, long pad_y, long b,
                     long old_height, long old_width) {
    // long d = 1;

    long k_cin = kernel[0];
    long k_cout = kernel[1];
    long k_h = kernel[2];
    long k_w = kernel[3];

    auto block_size = kernel;
    // clone because both matmul will run clone, so why not do it once
    long n = b;
    long c = cols.get_shape()[1];

    long h = old_height;
    long w = old_width;

    long sx = strides[0];
    long sy = strides[1];

    long dx = 1;  // later
    long dy = 1;

    long new_height = (old_height + 2 * 0 - k_h) / sy + 1;
    long new_width = (old_width + 2 * 0 - k_w) / sx + 1;

    auto img = zeros(TensorShape({n, c, old_height, old_width}), default_dtype);

    auto cols2 = sail::ops::reshape(
        cols,
        sail::TensorShape({new_height * new_width * b, k_cin * k_w * k_h}));

    // new_height = old_height;  // new_height * strides[0] - strides[0] + 3;
    // new_width = old_height;   // new_width * strides[1] - strides[1] + 3;
    long z = 0;
    int input_i = 0;
    int input_j = 0;
    // for (long k = 0; k < n; k++) {
    //     input_i = 0;

    for (long i = 0; i < new_height; i++) {

        input_j = 0;
        for (long j = 0; j < new_width; j++) {
            sail::Tensor col_slice = cols2.slice(sail::Slice({z, z + 1}));

            // std::cout << col_slice << std::endl;

            auto col_slice2 = col_slice.reshape(block_size);
            auto s = sail::Slice(
                {{}, {}, {input_i, k_h + input_i}, {input_j, input_j + k_w}});
            auto t = img.slice(s);
            img.slice(s).assign(t + col_slice2);
            z += 1;
            input_j += strides[1];
        }
        input_i += strides[0];
    }
    // }
    // auto img2 = img.slice(
    //     sail::Slice({{}, {}, {pad_y, nh - pad_y}, {pad_x, nw - pad_y}}));

    return img;

}

}  // namespace sail