

#pragma once

#include "loss.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../factories.h"
#include "../kernels/kernel.h"
#include "../ops/ops.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

std::string SoftmaxCrossEntropyLoss::getName() { return "MaxOp"; }
Tensor SoftmaxCrossEntropyLoss::forward(TensorVector inputs) {
    // stored_output = ops::softmax_cross_entropy(inputs[0], inputs[1]);
    return ops::softmax_cross_entropy(inputs[0], inputs[1]);
}
TensorVector SoftmaxCrossEntropyLoss::backward(Tensor& grad) {
    Tensor y = ops::softmax(Function::arg_storage[0]);
    Tensor z = empty(0, y.get_dtype(), y.get_shape());
    SoftmaxBackwardSubtractKernel().execute(y, Function::arg_storage[1], z);
    // std::cout << z << std::endl;
    return {z};
}

}  // namespace autograd
}  // namespace sail