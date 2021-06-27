
#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class AddMM : public Function {
   public:
    explicit AddMM(){};
    // TensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Matmul : public Function {
   public:
    std::string trans_a, trans_b;
    Matmul(std::string _trans_a, std::string _trans_b)
        : trans_a(_trans_a), trans_b(_trans_b){};
    // TensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail