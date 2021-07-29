#pragma once

#include "function.h"
#include <iostream>
#include <vector>
#include "Tensor.h"
#include "exception.h"
#include "factories.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

Tensor Function::apply(TensorVector& inputs) {
    COPY_INPUTS(inputs, arg_storage);
    DISABLE_GRAD(inputs);
    Tensor o = forward(inputs);
    ENABLE_GRAD(inputs);
    o.requires_grad = true;
    o.register_op(this);
    return o;
}

void Function::apply_no_forward(TensorVector& inputs) {
    COPY_INPUTS(inputs, arg_storage);
}
Tensor Function::set_fcn(Tensor& t) {
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