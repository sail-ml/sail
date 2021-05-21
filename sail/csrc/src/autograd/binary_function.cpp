#pragma once

#include "binary_function.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../factories.h"
#include "../ops/ops.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

/** begin block
 * cName = [Add, Subtract, Divide, Multiply]
 * op = [+, -, /, *]
 */

std::string Add::getName() { return "AddOp"; }
Tensor Add::forward(TensorVector inputs) {
    return ops::add(inputs[0], inputs[1]);
}
TensorVector Add::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(grad);
    return o;
}

std::string Subtract::getName() { return "SubtractOp"; }
Tensor Subtract::forward(TensorVector inputs) {
    return ops::subtract(inputs[0], inputs[1]);
}
TensorVector Subtract::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(-grad);
    return o;
}

std::string Divide::getName() { return "DivideOp"; }
Tensor Divide::forward(TensorVector inputs) {
    return ops::divide(inputs[0], inputs[1]);
}
TensorVector Divide::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];

    Tensor gx0 = grad / b;

    Tensor gx1 = -gx0 * a / b;  // * a;  //((a / b) / b);

    TensorVector o = {gx0, gx1};
    return o;
}

std::string Multiply::getName() { return "MultiplyOp"; }
Tensor Multiply::forward(TensorVector inputs) {
    return ops::multiply(inputs[0], inputs[1]);
}
TensorVector Multiply::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];
    TensorVector o = {b, a};
    return o;
}

std::string Matmul::getName() { return "MatmulOp"; }
Tensor Matmul::forward(TensorVector inputs) {
    return ops::matmul(inputs[0], inputs[1]);
}
TensorVector Matmul::backward(Tensor& grad) {
    Tensor a = Function::arg_storage[0];
    Tensor b = Function::arg_storage[1];
    TensorVector o = {b, a};
    return o;
}

}  // namespace autograd
}  // namespace sail