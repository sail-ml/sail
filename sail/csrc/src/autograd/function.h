#pragma once

#include <iostream>
#include <vector>
#include "../Tensor.h"

namespace sail {
// class Tensor {
//     bool requires_grad;
//     void register_op(Function* new_func);
// };  // let compiler know Tensor exists. Forward declaration

namespace autograd {

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

using TensorVector = std::vector<Tensor>;

class Function {
   public:
    TensorVector arg_storage;
    explicit Function(){};
    std::string name = "NONE";
    virtual std::string getName();
    virtual inline Tensor forward(TensorVector inputs);
    virtual inline Tensor apply(TensorVector inputs);
    virtual inline TensorVector backward(Tensor grad);
};

}  // namespace autograd
}  // namespace sail