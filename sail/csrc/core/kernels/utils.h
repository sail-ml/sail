// allow-no-source
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
#include "ops/CopyOps.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {

inline std::vector<std::vector<long>> calc_2d_same_padding(
    TensorShape kernel_shape) {
    int k_idx_1 = 2;
    int k_idx_2 = 3;

    if (kernel_shape.ndim() == 2) {
        k_idx_1 = 0;
        k_idx_2 = 1;
    }

    std::vector<long> padding_l, padding_r;
    long total_height_p = 1 * (kernel_shape[k_idx_1] - 1);
    long top_pad = total_height_p / 2;
    long bottom_pad = total_height_p - top_pad;

    long total_width_p = 1 * (kernel_shape[k_idx_2] - 1);
    long left_pad = total_width_p / 2;
    long right_pad = total_height_p - top_pad;

    padding_l.push_back(top_pad);
    padding_l.push_back(left_pad);

    padding_r.push_back(bottom_pad);
    padding_r.push_back(right_pad);

    return {padding_l, padding_r};
}

inline std::vector<long> calculate_nh_nw(TensorShape input_shape,
                                         TensorShape kernel_shape,
                                         std::vector<long> strides,
                                         std::string padding_mode) {
    long k_h, k_w;
    if (kernel_shape.ndim() == 4) {
        k_h = kernel_shape[2];
        k_w = kernel_shape[3];
    } else {
        k_h = kernel_shape[0];
        k_w = kernel_shape[1];
    }

    long new_height, new_width;

    if (padding_mode == "same") {
        new_height = input_shape[2];
        new_width = input_shape[3];
    } else {
        new_height = (input_shape[2] - (k_h)) / strides[0] + 1;
        new_width = (input_shape[3] - (k_w)) / strides[1] + 1;
    }

    return {new_height, new_width};
}

inline Tensor im2col(Tensor& im2col_input, TensorShape kernel_shape,
                     std::vector<long> strides, std::string padding_mode) {
    long d = 1;

    std::vector<long> padding_r;
    std::vector<long> padding_l;
    if (padding_mode == "same") {
        long total_height_p = 1 * (kernel_shape.shape[2] - 1);
        long top_pad = total_height_p / 2;
        long bottom_pad = total_height_p - top_pad;

        long total_width_p = 1 * (kernel_shape.shape[3] - 1);
        long left_pad = total_width_p / 2;
        long right_pad = total_height_p - top_pad;

        padding_l.push_back(top_pad);
        padding_l.push_back(left_pad);

        padding_r.push_back(bottom_pad);
        padding_r.push_back(right_pad);

    } else {
        padding_r = {0, 0};
        padding_l = {0, 0};
    }

    auto im2col_input2 = im2col_input;
    im2col_input2 =
        sail::ops::pad(im2col_input2, {{0, 0}, {0, 0}, padding_l, padding_r});

    std::vector<std::vector<long>> slices = {
        {},
        {},
        {0, im2col_input2.get_shape()[2], strides[0]},
        {0, im2col_input2.get_shape()[3], strides[1]}};

    std::vector<long> strides_ = {(long)1, 1, strides[0], strides[1]};
    auto strides_tensor = from_data((void*)strides_.data(), Dtype::sInt64,
                                    TensorShape({static_cast<long>(4)}));

    auto index_strides_ =
        im2col_input2.slice(Slice(slices)).get_shape().strides;

    long index_strides_size = static_cast<long>(index_strides_.size());
    long im2col_size = im2col_input2.ndim();
    long kernel_size = kernel_shape.ndim();

    auto window_shape = from_data((void*)kernel_shape.shape.data(),
                                  Dtype::sInt64, TensorShape({kernel_size}));

    auto window_strides =
        from_data((void*)im2col_input2.get_shape().strides.data(),
                  Dtype::sInt64, TensorShape({im2col_size}));
    auto indexing_strides =
        from_data((void*)index_strides_.data(), Dtype::sInt64,
                  TensorShape({index_strides_size}));
    auto in_shape = from_data((void*)im2col_input2.get_shape().shape.data(),
                              Dtype::sInt64, TensorShape({im2col_size}));

    auto win_indices_shape = ((in_shape - window_shape) / strides_tensor) + 1;
    auto c_win_indices_shape = win_indices_shape.cast(Dtype::sInt64);

    auto new_shape = sail::ops::cat({c_win_indices_shape, window_shape});
    auto new_strides = sail::ops::cat({indexing_strides, window_strides});

    std::vector<long> ns((long*)new_shape.get_data(),
                         (long*)new_shape.get_data() + new_shape.numel());
    std::vector<long> ns_((long*)new_strides.get_data(),
                          (long*)new_strides.get_data() + new_strides.numel());

    auto cols2 = as_strided(im2col_input2, sail::TensorShape(ns, ns_));

    return cols2;
}

inline Tensor col2im(Tensor& cols, TensorShape kernel,
                     std::vector<long> strides, long pad_x, long pad_y, long b,
                     long old_height, long old_width) {
    long k_cin = kernel[0];
    long k_h = kernel[2];
    long k_w = kernel[3];

    const auto& block_size = kernel;
    // clone because both matmul will run clone, so why not do it once
    long n = b;
    long c = cols.get_shape()[1];

    long h = old_height;
    long w = old_width;

    long sx = strides[0];
    long sy = strides[1];

    long dx = 1;
    long dy = 1;

    long new_height = (old_height + 2 * 0 - k_h) / sy + 1;
    long new_width = (old_width + 2 * 0 - k_w) / sx + 1;

    auto img = zeros(TensorShape({n, c, old_height, old_width}), default_dtype);

    auto cols2 = sail::ops::reshape(
        cols,
        sail::TensorShape({new_height * new_width * b, k_cin * k_w * k_h}));

    long z = 0;
    int input_i = 0;
    int input_j = 0;

    for (long i = 0; i < new_height; i++) {
        input_j = 0;
        for (long j = 0; j < new_width; j++) {
            sail::Tensor col_slice = cols2.slice(sail::Slice({z, z + 1}));

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

    return img;
}

}  // namespace sail