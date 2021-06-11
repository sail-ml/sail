
#pragma once

#include "reduction_function.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "factories.h"
#include "function.h"
#include "ops/ops.h"
#include "ops/reduction.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

std::string Sum::getName() { return "SumOp"; }
Tensor Sum::forward(TensorVector inputs) {
    return ops::sum(inputs[0], Reduction::axis, Reduction::keepdims);
}
TensorVector Sum::backward(Tensor& grad) {
    // if  (!(Function::arg_storage[0].get_ndim() == 0 || axis == -1 || keepdims
    // == true)) {
    //     grad = grad.expand_dims(axis);
    // }
    Tensor full_size =
        ops::broadcast_to(grad, Function::arg_storage[0].get_shape());
    return {full_size};
}

std::string Mean::getName() { return "MeanOp"; }
Tensor Mean::forward(TensorVector inputs) {
    return ops::mean(inputs[0], Reduction::axis, Reduction::keepdims);
}
TensorVector Mean::backward(Tensor& grad) {
    // if  (!(Function::arg_storage[0].get_ndim() == 0 || axis == -1 || keepdims
    // == true)) {
    //     grad = grad.expand_dims(axis);
    // }
    Tensor full_size =
        ops::broadcast_to(grad, Function::arg_storage[0].get_shape());
    return {full_size};
}

std::string Max::getName() { return "MaxOp"; }
Tensor Max::forward(TensorVector inputs) {
    stored_output = ops::max(inputs[0], Reduction::axis, Reduction::keepdims);
    return stored_output;
}
TensorVector Max::backward(Tensor& grad) {
    Tensor cond =
        ops::elementwise_equal(Function::arg_storage[0], stored_output);
    // if  (!(Function::arg_storage[0].get_ndim() == 0 || axis == -1 || keepdims
    // == true)) {
    //     grad = grad.expand_dims(axis);
    // }
    // Tensor full_size =
    //     ops::broadcast_to(grad, Function::arg_storage[0].get_shape());
    return {grad * cond};
}

}  // namespace autograd
}  // namespace sail