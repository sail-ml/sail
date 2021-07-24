

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
#include "onednn/conv2d_backward_data.h"
#include "onednn/conv2d_backward_weights.h"
#endif
using namespace std::chrono;

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

Tensor Conv2D::forward(TensorVector inputs) {
    auto output = ops::conv2d(inputs[0], inputs[1], strides, padding_mode);
    return output;
}
TensorVector Conv2D::backward(Tensor& grad) { THROW_ERROR(SailCError, "NOPE"); }

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