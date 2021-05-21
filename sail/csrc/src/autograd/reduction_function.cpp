
#pragma once

#include "reduction_function.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../factories.h"
#include "../ops/broadcast.h"
#include "../ops/reduction.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

std::string Sum::getName() { return "SumOp"; }
Tensor Sum::forward(TensorVector inputs) { return ops::sum(inputs[0]); }
TensorVector Sum::backward(Tensor& grad) {
    Tensor full_size =
        ops::broadcast_to(grad, Function::arg_storage[0].get_shape());
    return {clone(full_size)};
}

}  // namespace autograd
}  // namespace sail