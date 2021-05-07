#pragma once

#include <iostream>

#include "../Tensor.h"
#include "../dtypes.h"
#include "../factories.h"
#include "../kernels/kernel.h"

namespace sail {

namespace ops {

Tensor negate(Tensor& tensor1) {
    Tensor empty_tensor =
        empty(tensor1.ndim, tensor1.dtype, tensor1.shape_details);
    std::cout << "executing kernel" << std::endl;
    NegateTKernel().execute(tensor1, empty_tensor);
    return empty_tensor;
}

}  // namespace ops

}  // namespace sail
