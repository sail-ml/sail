

#pragma once

#include <iostream>
#include <vector>
#include "Tensor.h"
#include "function.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {

namespace autograd {

using TensorVector = std::vector<Tensor>;

class Reshape : public Function {
   public:
    TensorShape original_shape, new_shape;
    Reshape(TensorShape new_shape) : new_shape(new_shape){};
    Tensor forward(TensorVector inputs);
    TensorVector backward(Tensor& grad);

    ~Reshape(){};
};

}  // namespace autograd
}  // namespace sail