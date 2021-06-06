

#pragma once

#include "loss.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../factories.h"
#include "../ops/ops.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

std::string SoftmaxCrossEntropyLoss::getName() { return "MaxOp"; }
Tensor SoftmaxCrossEntropyLoss::forward(TensorVector inputs) {
    stored_output = ops::softmax_cross_entropy(inputs[0], inputs[1]);
    return stored_output;
}
TensorVector SoftmaxCrossEntropyLoss::backward(Tensor& grad) {
    Tensor y = ops::softmax(Function::arg_storage[0]);
    Tensor one_hot_tensor =
        one_hot(Function::arg_storage[1],
                Function::arg_storage[0].get_shape().shape[1]);
    Tensor casted_one_hot_tensor =
        ops::cast(one_hot_tensor, Function::arg_storage[0].get_dtype());
    return {y - casted_one_hot_tensor};
}

}  // namespace autograd
}  // namespace sail