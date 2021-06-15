
#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "function.h"
#include "ops/ops.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

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

class ClipMinOnly : public Function {
   public:
    Tensor stored_log;
    double min;
    explicit ClipMinOnly(double _min) : min(_min) {};
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Clip : public Function {
   public:
    Tensor stored_log;
    double min, max;
    explicit Clip(double _min, double _max) : min(_min), max(_max) {};
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail