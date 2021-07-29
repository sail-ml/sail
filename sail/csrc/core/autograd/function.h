#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"

#define DISABLE_GRAD_IND(i)                    \
    {                                          \
        i.was_requires_grad = i.requires_grad; \
        i.requires_grad = false;               \
    }
#define DISABLE_GRAD(inputs)     \
    {                            \
        for (auto& t : inputs) { \
            DISABLE_GRAD_IND(t)  \
        }                        \
    }
#define ENABLE_GRAD_IND(i) \
    { i.requires_grad = i.was_requires_grad; }
#define ENABLE_GRAD(inputs)      \
    {                            \
        for (auto& t : inputs) { \
            ENABLE_GRAD_IND(t);  \
        }                        \
    }
#define COPY_INPUTS(inputs, storage) \
    {                                \
        for (auto& t : inputs) {     \
            storage.push_back(t);    \
        }                            \
    }

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;
using RefTensorVector = std::vector<Tensor*>;

class Function {
   public:
    TensorVector arg_storage;
    TensorVector result_storage;
    explicit Function() = default;
    virtual Tensor forward(TensorVector inputs);
    virtual Tensor apply(TensorVector& inputs);
    virtual void apply_no_forward(TensorVector& inputs);
    virtual TensorVector backward(Tensor& grad);
    virtual Tensor set_fcn(Tensor& t);

    virtual ~Function() = default;
};

}  // namespace autograd
}  // namespace sail