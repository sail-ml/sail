

#pragma once

#include "ConvGrad.h"
#include <chrono>
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "autograd/function.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"
#ifdef MKLDNN
#include "onednn/conv2d.h"
#include "onednn/conv2d_backward_data.h"
#include "onednn/conv2d_backward_weights.h"
#endif
using namespace std::chrono;

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

Tensor Conv2D::forward(TensorVector inputs) {
    std::tuple<std::vector<Tensor>, long, long> tuple_output =
        ops::conv2d_impl(inputs[0], inputs[1], strides, padding_mode);
    kh = inputs[1].get_shape()[2];
    kw = inputs[1].get_shape()[3];
    auto output = std::get<0>(tuple_output);
    pad_y = std::get<1>(tuple_output);
    pad_x = std::get<2>(tuple_output);
    cols = output[1];
    flat_kernel = output[2];
    return output[0];
}
TensorVector Conv2D::backward(Tensor& grad) {
    auto block_size = Function::arg_storage[1].get_shape();
    // clone because both matmul will run clone, so why not do it once
    auto grad_ = clone(grad.reshape(TensorShape({grad.numel(), 1})));

    auto weight_grad = ops::matmul(cols, grad_, "T", "N");
    weight_grad._inplace_reshape(block_size);

    auto dcol = ops::matmul(grad_, flat_kernel, "N", "N");

    long n = Function::arg_storage[0].get_shape()[0];
    long c = Function::arg_storage[0].get_shape()[1];

    long h = Function::arg_storage[0].get_shape()[2];
    long w = Function::arg_storage[0].get_shape()[3];

    long sx = strides[0];
    long sy = strides[1];

    long dx = 1;  // later
    long dy = 1;

    long new_height = (h + 2 * pad_y - kh) / sy + 1;
    long new_width = (w + 2 * pad_x - kw) / sx + 1;

    long nh = h + 2 * pad_y + sy - 1;
    long nw = w + 2 * pad_x + sx - 1;
    auto img = zeros(TensorShape({n, c, nh, nw}), default_dtype);

    long z = 0;
    int input_i = 0;
    int input_j = 0;
    for (long k = 0; k < n; k++) {
        input_i = 0;

        for (long i = 0; i < new_height; i++) {
            input_j = 0;
            for (long j = 0; j < new_width; j++) {
                sail::Tensor col_slice = dcol.slice(sail::Slice({z, z + 1}));
                // std::cout << col_slice << std::endl;

                auto col_slice2 = col_slice.reshape(block_size);
                auto s = sail::Slice({{k, k + 1},
                                      {},
                                      {input_i, kh + input_i},
                                      {input_j, input_j + kw}});
                auto t = img.slice(s);
                img.slice(s).assign(t + col_slice2);
                z += 1;
                input_j += 1;
            }
            input_i += 1;
        }
    }
    auto img2 = img.slice(
        sail::Slice({{}, {}, {pad_y, nh - pad_y}, {pad_x, nw - pad_y}}));

    return {img2, weight_grad};
}

#ifdef MKLDNN
Tensor Conv2DMKLDNN::forward(TensorVector inputs) {
    THROW_ERROR_DETAILED(SailCError, "Should not call this function");
}
TensorVector Conv2DMKLDNN::backward(Tensor& grad) {
    if (grad.is_view()) {
        grad = clone(grad);
    }
    Tensor input, weights, biases;
    bool use_bias = true;
    if (Function::arg_storage.size() == 3) {
        input = Function::arg_storage[0];
        weights = Function::arg_storage[1];
        biases = Function::arg_storage[2];
    } else {
        input = Function::arg_storage[0];
        weights = Function::arg_storage[1];
        biases = weights;
        use_bias = false;
    }
    Tensor kernel_grad = empty(0, Dtype::sFloat32, weights.get_shape());
    Tensor bias_grad = empty(0, Dtype::sFloat32, biases.get_shape());
    Tensor src_grad = empty(0, Dtype::sFloat32, input.get_shape());

    if (use_bias) {
        auto L = onednn::Conv2DBackwardWeightsFactory(
            input, kernel_grad, bias_grad, grad, strides, padding_l, padding_r);
        auto L2 = onednn::Conv2DBackwardDataFactory(
            src_grad, weights, grad, strides, padding_l, padding_r);

        L.forward();
        L2.forward();

        return {src_grad, kernel_grad, bias_grad};
    } else {
        auto L = onednn::Conv2DBackwardWeightsFactory(
            input, kernel_grad, grad, strides, padding_l, padding_r);
        auto L2 = onednn::Conv2DBackwardDataFactory(
            src_grad, weights, grad, strides, padding_l, padding_r);
        L.forward();
        L2.forward();
        return {src_grad, kernel_grad};
    }
}

#endif

}  // namespace autograd
}  // namespace sail