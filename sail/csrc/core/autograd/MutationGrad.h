

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
    Reshape(TensorShape new_shape) : new_shape(std::move(new_shape)){};
    Tensor forward(TensorVector inputs) override;
    TensorVector backward(Tensor& grad) override;

    ~Reshape() override = default;
};

}  // namespace autograd
}  // namespace sail