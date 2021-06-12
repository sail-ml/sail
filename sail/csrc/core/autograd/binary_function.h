
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
    explicit Add(){};
    // TensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Subtract : public Function {
   public:
    explicit Subtract(){};
    // TensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Divide : public Function {
   public:
    explicit Divide(){};
    // TensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Multiply : public Function {
   public:
    explicit Multiply(){};
    // TensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Matmul : public Function {
   public:
    std::string trans_a, trans_b;
    Matmul(std::string _trans_a, std::string _trans_b)
        : trans_a(_trans_a), trans_b(_trans_b){};
    // TensorVector arg_storage;
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

class Pow : public Function {
   public:
    explicit Pow(){};
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail