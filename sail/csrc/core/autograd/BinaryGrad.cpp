#pragma once

#include "BinaryGrad.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "factories.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

Tensor Add::forward(TensorVector inputs) {
    return ops::add(inputs[0], inputs[1]);
}
TensorVector Add::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(grad);
    return o;
}

Tensor Subtract::forward(TensorVector inputs) {
    return ops::subtract(inputs[0], inputs[1]);
}
TensorVector Subtract::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(-grad);
    return o;
}

Tensor Divide::forward(TensorVector inputs) {
    return ops::divide(inputs[0], inputs[1]);
}
TensorVector Divide::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];

    Tensor gx0 = grad / b;

    Tensor gx1 = -gx0 * a / b;

    TensorVector o = {gx0, gx1};
    return o;
}

Tensor Multiply::forward(TensorVector inputs) {
    return ops::multiply(inputs[0], inputs[1]);
}
TensorVector Multiply::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];
    TensorVector o = {b * grad, a * grad};
    return o;
}

}  // namespace autograd
}  // namespace sail