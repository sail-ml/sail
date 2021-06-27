
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
    explicit Pow(){};
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Exp : public Function {
   public:
    explicit Exp(){};
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Log : public Function {
   public:
    Tensor stored_log;
    explicit Log(){};
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail