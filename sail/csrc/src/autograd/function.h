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

using TensorVector = std::vector<Tensor>;
using RefTensorVector = std::vector<Tensor*>;
// using RefTensorVector = std::vector<Tensor*>;

class Function {
   public:
    RefTensorVector arg_storage;
    explicit Function(){};
    std::string name = "NONE";
    virtual std::string getName();
    virtual inline Tensor forward(RefTensorVector inputs);
    virtual inline Tensor apply(RefTensorVector& inputs);
    virtual inline TensorVector backward(Tensor grad);
};

}  // namespace autograd
}  // namespace sail