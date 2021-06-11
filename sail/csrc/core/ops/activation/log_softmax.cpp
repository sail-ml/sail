#pragma once

#include <iostream>

#include <chrono>
#include "Tensor.h"
#include "autograd/autograd.h"
#include "factories.h"
#include "kernels/kernel.h"
#include "ops/ops.h"

#include "softmax.h"
using namespace std::chrono;

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor log_softmax(Tensor& input, int axis = 1) {
    if (input.requires_grad) {
        TensorVector vec;
        vec.emplace_back(input);
        return (new autograd::Softmax(axis))->apply(vec);
    }

    Tensor max = ops::max(input, axis, true);
    input = input - max;
    input = ops::exp(input);
    Tensor s = ops::sum(input, axis, true);
    input = input - ops::log(s);
    return input;
}

}  // namespace ops
}  // namespace sail
