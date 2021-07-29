#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class Add : public Function {
   public:
    explicit Add() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Subtract : public Function {
   public:
    explicit Subtract() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Divide : public Function {
   public:
    explicit Divide() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Multiply : public Function {
   public:
    explicit Multiply() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

}  // namespace autograd
}  // namespace sail