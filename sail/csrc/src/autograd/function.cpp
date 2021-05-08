#pragma once

#include "function.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"

namespace sail {

namespace autograd {

#define DISABLE_GRAD(inputs)                      \
    {                                             \
        for (int i = 0; i < inputs.size(); i++) { \
            inputs[i]->requires_grad = false;     \
        }                                         \
    }
#define ENABLE_GRAD(inputs)                       \
    {                                             \
        for (int i = 0; i < inputs.size(); i++) { \
            inputs[i]->requires_grad = true;      \
        }                                         \
    }

using TensorVector = std::vector<Tensor>;

std::string Function::getName() { return "NONE"; }
inline Tensor Function::apply(RefTensorVector& inputs) {
    arg_storage = inputs;
    DISABLE_GRAD(inputs);
    Tensor o = forward(inputs);
    ENABLE_GRAD(inputs);
    o.requires_grad = true;
    o.register_op(this);
    return o;
}
inline Tensor Function::forward(RefTensorVector inputs) {
    throw "not implemented yet.";
}
inline TensorVector Function::backward(Tensor inputs) {
    throw "not implemented yet.";
}

}  // namespace autograd
}  // namespace sail