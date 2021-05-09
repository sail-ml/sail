#pragma once

#include "binary_function.h"
#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../ops/elementwise.h"
#include "function.h"

#define EXECUTE_OP(a, b, o, op)  \
    {                            \
        a.requires_grad = false; \
        b.requires_grad = false; \
        o = op(a, b);            \
        a.requires_grad = true;  \
        b.requires_grad = true;  \
        o.requires_grad = true;  \
    }
#define DISABLE_GRAD(inputs)         \
    {                                \
        for (Tensor t : inputs) {    \
            t.requires_grad = false; \
        }                            \
    }
#define ENABLE_GRAD(inputs)         \
    {                               \
        for (Tensor t : inputs) {   \
            t.requires_grad = true; \
        }                           \
    }

#define APPLY(inputs)               \
    {                               \
        arg_storage = inputs;       \
        DISABLE_GRAD(inputs);       \
        Tensor o = forward(inputs); \
        o.requires_grad = true;     \
        o.register_op(this);        \
        return o;                   \
    }

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

/** begin block
 * cName = [Add, Subtract, Divide, Multiply]
 * op = [+, -, /, *]
 */

std::string Add::getName() { return "AddOp"; }
inline Tensor Add::forward(RefTensorVector inputs) {
    return ops::add(*inputs[0], *inputs[1]);
}
inline TensorVector Add::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(grad);
    return o;
}

std::string Subtract::getName() { return "SubtractOp"; }
inline Tensor Subtract::forward(RefTensorVector inputs) {
    return ops::subtract(*inputs[0], *inputs[1]);
}
inline TensorVector Subtract::backward(Tensor& grad) {
    TensorVector o;
    o.emplace_back(grad);
    o.emplace_back(-grad);
    return o;
}

std::string Divide::getName() { return "DivideOp"; }
inline Tensor Divide::forward(RefTensorVector inputs) {
    return ops::divide(*inputs[0], *inputs[1]);
}
inline TensorVector Divide::backward(Tensor& grad) {
    Tensor a = *Function::arg_storage[0];
    Tensor b = *Function::arg_storage[1];

    Tensor gx0 = grad / b;
    Tensor gx1 = (-grad) * ((a / b) / b);
    // TensorVector o = {grad * (*b), grad * (*a)};
    TensorVector o;  // = {b, a};
    o.emplace_back(gx0);
    o.emplace_back(gx1);
    return o;
}

std::string Multiply::getName() { return "MultiplyOp"; }
inline Tensor Multiply::forward(RefTensorVector inputs) {
    return ops::multiply(*inputs[0], *inputs[1]);
}
inline TensorVector Multiply::backward(Tensor& grad) {
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