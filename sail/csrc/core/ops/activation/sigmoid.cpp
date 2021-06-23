#pragma once

#include <iostream>

#include "Tensor.h"
#include "autograd/autograd.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"

#include "sigmoid.h"

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor sigmoid(Tensor& input) {
    if (input.requires_grad) {
        TensorVector vec;
        vec.emplace_back(input);
        Tensor e = (new autograd::Sigmoid())->apply(vec);
        return e;
    }
    Tensor empty_tensor = empty_like(input);
    sail::internal::sigmoid_stub(
        input, empty_tensor);  // SigmoidKernel().execute(input, empty_tensor);
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
