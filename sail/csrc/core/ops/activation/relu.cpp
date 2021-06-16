#pragma once

#include <iostream>

#include "Tensor.h"
#include "autograd/autograd.h"
#include "factories.h"
#include "kernels/kernel.h"
#include "ops/ops.h"

#include "relu.h"

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor ReLU(Tensor& input) {
    Tensor empty_tensor;
    if (input.requires_grad) {
        TensorVector vec;
        vec.emplace_back(input);
        empty_tensor = (new autograd::ReLU())->apply(vec);
        return empty_tensor;
    }

    empty_tensor = empty_like(input);
    ClipMinOnlyKernel().execute(input, 0.0, empty_tensor);
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
