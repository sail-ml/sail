
#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class Pow : public Function {
   public:
    explicit Pow() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Exp : public Function {
   public:
    explicit Exp() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Log : public Function {
   public:
    Tensor stored_log;
    explicit Log() = default;
    ~Log() override = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

}  // namespace autograd
}  // namespace sail