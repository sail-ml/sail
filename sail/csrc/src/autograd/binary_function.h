
#pragma once

#include <iostream>
#include <vector>
#include "../Tensor.h"
#include "../ops/elementwise.h"
#include "function.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class Add : public Function {
   public:
    explicit Add(){};
    // TensorVector arg_storage;
    std::string getName();
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Subtract : public Function {
   public:
    explicit Subtract(){};
    // TensorVector arg_storage;
    std::string getName();
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Divide : public Function {
   public:
    explicit Divide(){};
    // TensorVector arg_storage;
    std::string getName();
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Multiply : public Function {
   public:
    explicit Multiply(){};
    // TensorVector arg_storage;
    std::string getName();
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail