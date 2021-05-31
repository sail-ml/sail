
#pragma once

#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../ops/ops.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class AddMM : public Function {
   public:
    explicit AddMM(){};
    // TensorVector arg_storage;
    std::string getName();
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail