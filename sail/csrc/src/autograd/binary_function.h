
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
    // RefTensorVector arg_storage;
    std::string getName();
    Tensor forward(RefTensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Subtract : public Function {
   public:
    explicit Subtract(){};
    // RefTensorVector arg_storage;
    std::string getName();
    Tensor forward(RefTensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Divide : public Function {
   public:
    explicit Divide(){};
    // RefTensorVector arg_storage;
    std::string getName();
    Tensor forward(RefTensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Multiply : public Function {
   public:
    explicit Multiply(){};
    // RefTensorVector arg_storage;
    std::string getName();
    Tensor forward(RefTensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail