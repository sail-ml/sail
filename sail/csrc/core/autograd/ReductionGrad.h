
#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "function.h"
#include "ops/ops.h"

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
    std::vector<long> axis = {NULLDIM};
    bool keepdims = false;
    long numel = 1;
    Reduction(std::vector<long> axis, bool keepdims)
        : axis(std::move(axis)), keepdims(keepdims){};
    Reduction(std::vector<long> axis, bool keepdims, long numel)
        : axis(std::move(axis)), keepdims(keepdims), numel(numel){};
};

class Sum : public Reduction {
   public:
    using Reduction::Reduction;

    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};
class Mean : public Reduction {
   public:
    using Reduction::Reduction;

    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Max : public Reduction {
   public:
    using Reduction::Reduction;
    Tensor stored_output;

    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};
class Min : public Reduction {
   public:
    using Reduction::Reduction;
    Tensor stored_output;

    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

}  // namespace autograd
}  // namespace sail