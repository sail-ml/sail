
#include "activations.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "error.h"
#include "factories.h"
#include "function.h"
#include "kernels/kernel.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

std::string Sigmoid::getName() { return "SigmoidOp"; }
Tensor Sigmoid::forward(TensorVector inputs) {
    Tensor return_ = ops::sigmoid(inputs[0]);
    Function::result_storage.push_back(Tensor(return_.get_body(), false));
    return return_;  // Tensor(sigmoid_stored);
}
TensorVector Sigmoid::backward(Tensor& grad) {
    Tensor stored = Function::result_storage[0];
    Tensor result_tensor = empty(0, stored.get_dtype(), stored.get_shape());
    SigmoidBackwardKernel().execute(stored, result_tensor);
    return {grad * result_tensor};
    // return {grad *
    //         (stored * (ones(stored.get_shape(), stored.get_dtype()) -
    //         stored))};
}

std::string Softmax::getName() { return "SoftmaxOp"; }
Tensor Softmax::forward(TensorVector inputs) {
    Tensor result = ops::softmax(inputs[0], axis);
    Function::result_storage.push_back(Tensor(result));

    return result;
}
TensorVector Softmax::backward(Tensor& grad) {
    Tensor stored = Function::result_storage[0];
    Tensor new_grad = grad * stored;
    Tensor sum_new_grad = ops::sum(new_grad, axis, true);
    Tensor sub = grad - sum_new_grad;
    Tensor ret = sub * stored;
    return {ret};
}

}  // namespace autograd
}  // namespace sail