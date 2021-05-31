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
    Tensor empty_tensor = empty_like(input);
    SigmoidKernel().execute(input, empty_tensor);
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
