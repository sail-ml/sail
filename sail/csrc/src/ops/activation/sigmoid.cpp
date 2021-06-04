#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../autograd/autograd.h"
#include "../../factories.h"
#include "../../kernels/kernel.h"
#include "../../ops/ops.h"

#include "sigmoid.h"

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor sigmoid(Tensor& input) {
    if (input.requires_grad) {
        TensorVector vec;
        vec.emplace_back(input);
        return (new autograd::Sigmoid())->apply(vec);
    }
    Tensor empty_tensor = empty_like(input);
    SigmoidKernel().execute(input, empty_tensor);
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
