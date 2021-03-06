
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
    explicit SoftmaxCrossEntropyLoss() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class MeanSquaredErrorLoss : public Function {
   public:
    Tensor stored_output;
    explicit MeanSquaredErrorLoss() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

}  // namespace autograd
}  // namespace sail