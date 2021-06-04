#include "unary_function.h"
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

std::string Exp::getName() { return "ExpOp"; }
Tensor Exp::forward(TensorVector inputs) { return ops::exp(inputs[0]); }
TensorVector Exp::backward(Tensor& grad) {
    Tensor t = ops::exp(Function::arg_storage[0]);
    Tensor b = t * grad;
    return {b};
}

}  // namespace autograd
}  // namespace sail