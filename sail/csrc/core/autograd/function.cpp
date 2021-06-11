#pragma once

#include "function.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "error.h"
#include "factories.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

std::string Function::getName() { return "NONE"; }
Tensor Function::apply(TensorVector& inputs) {
    // arg_storage = inputs;
    // arg_storage(inputs);
    COPY_INPUTS(inputs, arg_storage);
    DISABLE_GRAD(inputs);
    Tensor o = forward(inputs);
    ENABLE_GRAD(inputs);
    o.requires_grad = true;
    o.register_op(this);
    return o;
}

void Function::apply_no_forward(TensorVector& inputs) {
    // arg_storage = inputs;
    COPY_INPUTS(inputs, arg_storage);
    // arg_storage(inputs);
}
Tensor Function::set_fcn(Tensor& t) {
    // arg_storage = inputs;
    t.requires_grad = true;
    t.register_op(this);
    return t;
}
Tensor Function::forward(TensorVector inputs) {
    throw SailCError("not implemented yet.");
}
TensorVector Function::backward(Tensor& inputs) {
    throw SailCError("not implemented yet.");
}

}  // namespace autograd
}  // namespace sail