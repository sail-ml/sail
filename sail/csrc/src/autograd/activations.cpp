
#include "activations.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../error.h"
#include "../factories.h"
#include "../ops/ops.h"
#include "../tensor_shape.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

std::string Sigmoid::getName() { return "SigmoidOp"; }
Tensor Sigmoid::forward(TensorVector inputs) {
    sigmoid_stored = ops::sigmoid(inputs[0]);
    return sigmoid_stored;
}
TensorVector Sigmoid::backward(Tensor& grad) {
    return {sigmoid_stored *
            (ones(sigmoid_stored.get_shape(), sigmoid_stored.get_dtype()) -
             sigmoid_stored)};
}

std::string Softmax::getName() { return "SoftmaxOp"; }
Tensor Softmax::forward(TensorVector inputs) {
    softmax_stored = ops::softmax(inputs[0], axis);
    return softmax_stored;
}
TensorVector Softmax::backward(Tensor& grad) {
    Tensor new_grad = grad * softmax_stored;
    Tensor sum_new_grad = ops::sum(new_grad, axis, true);
    Tensor sub = grad - sum_new_grad;
    Tensor ret = sub * softmax_stored;
    return {ret};
}

}  // namespace autograd
}  // namespace sail