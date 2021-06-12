

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
    explicit Sigmoid(){};
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};
class Softmax : public Function {
   public:
    Tensor softmax_stored;
    int axis;
    Softmax(int _axis = 1) { axis = _axis; };
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);
};

}  // namespace autograd
}  // namespace sail