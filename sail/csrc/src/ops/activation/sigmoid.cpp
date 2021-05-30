#pragma once

#include <iostream>

#include "../../Tensor.h"
#include "../../factories.h"
#include "../../kernels/kernel.h"
#include "../../ops/ops.h"
#include "sigmoid.h"

namespace sail {
namespace ops {

Tensor sigmoid(Tensor& input) {
    Tensor one = one_scalar(input.get_dtype());
    Tensor neg = -input;
    Tensor exp_ = exp(neg);
    Tensor base = one + exp_;
    Tensor final_ = one / base;
    return final_;
}

}  // namespace ops
}  // namespace sail
