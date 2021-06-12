
#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class SoftmaxCrossEntropyLoss : public Function {
   public:
    Tensor stored_output;
    explicit SoftmaxCrossEntropyLoss(){};
    // RefTensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail