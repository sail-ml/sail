
#include "activations.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../error.h"
#include "../factories.h"
#include "../ops/ops.h"
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

}  // namespace autograd
}  // namespace sail