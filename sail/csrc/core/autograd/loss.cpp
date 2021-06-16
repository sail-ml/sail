

#pragma once

#include "loss.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "factories.h"
#include "function.h"
#include "kernels/kernel.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

Tensor SoftmaxCrossEntropyLoss::forward(TensorVector inputs) {
    return ops::softmax_cross_entropy(inputs[0], inputs[1]);
}
TensorVector SoftmaxCrossEntropyLoss::backward(Tensor& grad) {
    Tensor y = ops::softmax(Function::arg_storage[0]);
    Tensor z = empty(0, y.get_dtype(), y.get_shape());
    SoftmaxBackwardSubtractKernel().execute(y, Function::arg_storage[1], z);
    return {z};
}

Tensor MeanSquaredErrorLoss::forward(TensorVector inputs) {
    return ops::mean_squared_error(inputs[0], inputs[1]);
}
TensorVector MeanSquaredErrorLoss::backward(Tensor& grad) {
    Tensor emp = empty_like(Function::arg_storage[0]);
    Tensor emp2 = empty_like(Function::arg_storage[0]);

    MeanSquaredErrorBackwardKernel().execute(
        Function::arg_storage[0],
        Function::arg_storage[1],
        grad,
        emp,
        emp2
    );

    return {emp, emp2};

}

}  // namespace autograd
}  // namespace sail