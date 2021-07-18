
#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class ClipMinOnly : public Function {
   public:
    Tensor stored_log;
    double min;
    explicit ClipMinOnly(double _min) : min(_min){};
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Clip : public Function {
   public:
    Tensor stored_log;
    double min, max;
    explicit Clip(double _min, double _max) : min(_min), max(_max){};
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

}  // namespace autograd
}  // namespace sail