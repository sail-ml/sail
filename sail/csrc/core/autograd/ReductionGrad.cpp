
#pragma once

#include "ReductionGrad.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "factories.h"
#include "function.h"
#include "ops/ReductionOps.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

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

Tensor Max::forward(TensorVector inputs) {
    stored_output = ops::max(inputs[0], Reduction::axis, Reduction::keepdims);
    return stored_output;
}
TensorVector Max::backward(Tensor& grad) {
    Tensor cond =
        ops::elementwise_equal(Function::arg_storage[0], stored_output);
    return {grad * cond};
}

Tensor Min::forward(TensorVector inputs) {
    stored_output = ops::min(inputs[0], Reduction::axis, Reduction::keepdims);
    return stored_output;
}
TensorVector Min::backward(Tensor& grad) {
    Tensor cond =
        ops::elementwise_equal(Function::arg_storage[0], stored_output);
    return {grad * cond};
}

}  // namespace autograd
}  // namespace sail