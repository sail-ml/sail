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
Tensor Add::forward(RefTensorVector inputs) {
    return ops::add(*inputs[0], *inputs[1]);
}
TensorVector Add::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(grad);
    return o;
}

std::string Subtract::getName() { return "SubtractOp"; }
Tensor Subtract::forward(RefTensorVector inputs) {
    return ops::subtract(*inputs[0], *inputs[1]);
}
TensorVector Subtract::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(-grad);
    return o;
}

std::string Divide::getName() { return "DivideOp"; }
Tensor Divide::forward(RefTensorVector inputs) {
    return ops::divide(*inputs[0], *inputs[1]);
}
TensorVector Divide::backward(Tensor& grad) {
    Tensor a = *Function::arg_storage[0];
    Tensor b = *Function::arg_storage[1];

    // std::cout << "a.get_shape().get_string()" << std::endl;
    // std::cout << a.get_shape().get_string() << std::endl;

    // a.requires_grad = false;

    Tensor gx0 = ops::add(a, a);  // / b;

    Tensor gx1 = grad;  // * a;  //((a / b) / b);

    TensorVector o = {gx0, gx1};
    return o;
}

std::string Multiply::getName() { return "MultiplyOp"; }
Tensor Multiply::forward(RefTensorVector inputs) {
    return ops::multiply(*inputs[0], *inputs[1]);
}
TensorVector Multiply::backward(Tensor& grad) {
    Tensor a = *Function::arg_storage[0];
    Tensor b = *Function::arg_storage[1];
    TensorVector o = {b, a};
    // o.emplace_back(b);
    // o.emplace_back(a);
    return o;
}

/** end block **/

}  // namespace autograd
}  // namespace sail