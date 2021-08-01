

#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "TensorBody.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class Sigmoid : public Function {
   public:
    TensorBody::pointer sigmoid_stored;
    explicit Sigmoid() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Softmax : public Function {
   public:
    Tensor softmax_stored;
    int axis;
    Softmax(int axis = 1) : axis(axis){};
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class ReLU : public Function {
   public:
    explicit ReLU() = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

class Tanh : public Function {
   public:
    Tensor storage;
    explicit Tanh() = default;
    ~Tanh() override = default;
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;
};

}  // namespace autograd
}  // namespace sail