
#pragma once

#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../ops/ops.h"
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

class Reduction : public Function {
   public:
    int axis;
    bool keepdims;
    Reduction(int _axis, bool _keepdims) {
        axis = _axis;
        keepdims = _keepdims;
    };
};

class Sum : public Reduction {
   public:
    using Reduction::Reduction;

    // RefTensorVector arg_storage;
    std::string getName();
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Max : public Reduction {
   public:
    using Reduction::Reduction;
    Tensor stored_output;

    // RefTensorVector arg_storage;
    std::string getName();
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail