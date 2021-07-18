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
    std::string padding_mode) {
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

    long new_height = (h + 2 * 0 - 1 * (k_h - 1) - 1) / stride[0] + 1;
    long new_width = (w + 2 * 0 - 1 * (k_w - 1) - 1) / stride[1] + 1;

    auto cols2 = sail::im2col(im2col_input, kernel.get_shape(), stride, pad_x,
                              pad_y, b, new_height, new_width);

    auto cols = sail::ops::reshape(
        cols2,
        sail::TensorShape({new_height * new_width * b, k_cin * k_w * k_h}));

    // std::cout << duration.count() << std::endl;
    Tensor flat_kernel =
        kernel.reshape(TensorShape({k_cout, k_cin * k_w * k_h}));

    Tensor res = ops::matmul(cols, flat_kernel, "N", "T");
    res._inplace_reshape(TensorShape({b, k_cout, new_height, new_width}));

    std::vector<Tensor> first_vec = {res, cols, flat_kernel};

    return std::make_tuple(first_vec, pad_y, pad_x);
}

Tensor conv2d(Tensor& input, Tensor& kernel, std::vector<long> stride,
              std::string padding_mode) {
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
