#pragma once

#include <iostream>
#include <vector>
#include "../Tensor.h"

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
#define COPY_INPUTS(inputs, storage)              \
    {                                             \
        for (int i = 0; i < inputs.size(); i++) { \
            storage.push_back(inputs[i]);         \
        }                                         \
    }

namespace sail {
// class Tensor {
//     bool requires_grad;
//     void register_op(Function* new_func);
// };  // let compiler know Tensor exists. Forward declaration

namespace autograd {

using TensorVector = std::vector<Tensor>;
using RefTensorVector = std::vector<Tensor*>;
// using RefTensorVector = std::vector<Tensor*>;

class Function {
   public:
    RefTensorVector arg_storage;
    explicit Function(){};
    std::string name = "NONE";
    virtual std::string getName();
    virtual Tensor forward(RefTensorVector inputs);
    virtual Tensor apply(RefTensorVector& inputs);
    virtual TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail