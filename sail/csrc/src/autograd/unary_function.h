
#pragma once

#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../ops/ops.h"
#include "function.h"


namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class Exp : public Function {
   public:
    explicit Exp(){};
    std::string getName();
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail