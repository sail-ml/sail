#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../autograd/autograd.h"
#include "../../factories.h"
#include "../../kernels/kernel.h"
#include "../../ops/ops.h"

#include "softmax.h"

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor softmax(Tensor& input, int axis = 1) {
    if (input.requires_grad) {
        TensorVector vec;
        vec.emplace_back(input);
        return (new autograd::Softmax(axis))->apply(vec);
    }

    Tensor max = ops::max(input, axis, true);
    Tensor y = input - max;
    y = ops::exp(y);
    y = y / ops::sum(y, axis, true);
    return y;
}

}  // namespace ops
}  // namespace sail
