
#pragma once

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

class Add : public Function {
   public:
    explicit Add(){};
    TensorVector arg_storage;
    std::string getName();
    inline Tensor forward(TensorVector inputs);
};

class Subtract : public Function {
   public:
    explicit Subtract(){};
    TensorVector arg_storage;
    std::string getName();
    inline Tensor forward(TensorVector inputs);
};

class Divide : public Function {
   public:
    explicit Divide(){};
    TensorVector arg_storage;
    std::string getName();
    inline Tensor forward(TensorVector inputs);
};

class Multiply : public Function {
   public:
    explicit Multiply(){};
    TensorVector arg_storage;
    std::string getName();
    inline Tensor forward(TensorVector inputs);
};

}  // namespace autograd
}  // namespace sail