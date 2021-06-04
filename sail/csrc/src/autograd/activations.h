

#pragma once

#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../ops/ops.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class Sigmoid : public Function {
   public:
    Tensor sigmoid_stored;
    explicit Sigmoid(){};
    std::string getName();
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail