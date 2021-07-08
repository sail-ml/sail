#include <iostream>

#include "Tensor.h"
#include "autograd/autograd.h"
#include "dtypes.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"
#include "slice.h"
#include "tensor_shape.h"

#include <chrono>
using namespace std::chrono;
namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

std::tuple<std::vector<Tensor>, long, long> conv2d_impl(
    Tensor& input, Tensor& kernel, std::vector<long> stride,
    std::string padding_mode = "same") {
    // Tensor conv2d(Tensor& input, Tensor& kernel, std::vector<long> stride,
    //               std::string padding_mode = "same") {

    SAIL_CHECK(input.get_ndim() == 4,
               "Input to Conv2D must have ndim == 4, [NCHW]")
    SAIL_CHECK(
        kernel.get_ndim() == 4,
        "Kernel input to Conv2D must have ndim == 4, [C out, C in, H, W]")

    long d = 1;

    long pad_y = 0;
    long pad_x = 0;

    long b = input.get_shape()[0];
    long c = input.get_shape()[1];
    long h = input.get_shape()[2];
    long w = input.get_shape()[3];

    long k_cin = kernel.get_shape()[0];
    long k_cout = kernel.get_shape()[1];
    long k_h = kernel.get_shape()[2];
    long k_w = kernel.get_shape()[3];

    SAIL_CHECK_LINE(k_cin == c);

    Tensor im2col_input = input;
    if (padding_mode == "same") {
        pad_y =
            (long)(((1 - (float)d - (float)stride[0] + (float)k_h * (float)d) /
                    2) +
                   (float)h * ((-1 + (float)stride[0]) / 2));
        pad_x =
            (long)(((1 - (float)d - (float)stride[1] + (float)k_w * (float)d) /
                    2) +
                   (float)w * ((-1 + (float)stride[1]) / 2));
        im2col_input = ops::pad(
            im2col_input, {{0, 0}, {0, 0}, {pad_x, pad_y}, {pad_x, pad_y}});
    }

    long new_height = (h + 2 * pad_y - k_h) / stride[0] + 1;
    long new_width = (w + 2 * pad_x - k_w) / stride[1] + 1;

    std::vector<std::vector<long>> slices = {
        {},
        {},
        {0, im2col_input.get_shape()[2], stride[0]},
        {0, im2col_input.get_shape()[3], stride[1]}};

    std::vector<long> strides_ = {(long)1, 1, stride[0], stride[1]};
    auto strides_tensor = from_data((void*)strides_.data(), Dtype::sInt64,
                                    TensorShape({strides_.size()}));

    auto start = high_resolution_clock::now();

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
    auto c_win_indices_shape = ops::cast(win_indices_shape, Dtype::sInt64);

    auto new_shape = sail::ops::cat({c_win_indices_shape, window_shape});
    auto new_strides = sail::ops::cat({indexing_strides, window_strides});

    std::vector<long> ns((long*)new_shape.get_data(),
                         (long*)new_shape.get_data() + new_shape.numel());
    std::vector<long> ns_((long*)new_strides.get_data(),
                          (long*)new_strides.get_data() + new_strides.numel());

    auto cols2 = as_strided(im2col_input, sail::TensorShape(ns, ns_));

    auto cols = ops::reshape(
        cols2,
        sail::TensorShape({new_height * new_width * b, k_cin * k_w * k_h}));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    // std::cout << duration.count() << std::endl;
    Tensor flat_kernel =
        kernel.reshape(TensorShape({k_cout, k_cin * k_w * k_h}));

    Tensor res = ops::matmul(cols, flat_kernel, "N", "T");
    res._inplace_reshape(TensorShape({b, k_cout, new_height, new_width}));

    std::vector<Tensor> first_vec = {res, cols, flat_kernel};

    return std::make_tuple(first_vec, pad_y, pad_x);
}

Tensor conv2d(Tensor& input, Tensor& kernel, std::vector<long> stride,
              std::string padding_mode = "same") {
    if (input.requires_grad || kernel.requires_grad) {
        TensorVector vec;
        vec.emplace_back(input);
        vec.emplace_back(kernel);
        Tensor empty_tensor =
            (new autograd::Conv2D(stride, padding_mode))->apply(vec);

        return empty_tensor;
    }
    std::tuple<std::vector<Tensor>, long, long> ret =
        conv2d_impl(input, kernel, stride, padding_mode);
    return std::get<0>(ret)[0];
}

}  // namespace ops

}  // namespace sail
