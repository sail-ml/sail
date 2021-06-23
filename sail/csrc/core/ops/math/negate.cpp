#pragma once

#include <iostream>

#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/Kernel.h"

namespace sail {

namespace ops {

Tensor negate(Tensor& tensor1) {
    Tensor empty_tensor =
        empty(tensor1.get_ndim(), tensor1.get_dtype(), tensor1.get_shape());
    sail::internal::negate_stub(tensor1, empty_tensor);
    return empty_tensor;
}

}  // namespace ops

}  // namespace sail
