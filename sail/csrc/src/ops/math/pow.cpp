#pragma once

#include <iostream>

#include "../Tensor.h"
#include "../dtypes.h"
#include "../kernels/kernel.h"

namespace sail {

namespace ops {

Tensor pow(Tensor& tensor1, const double power) {
    Tensor empty_tensor = empty_like(tensor1);
    PowerKernel().execute(tensor1, power, empty_tensor);
    return empty_tensor;
}
Tensor exp(Tensor& tensor1) {
    Tensor empty_tensor = empty_like(tensor1);
    PowerExpKernel().execute(tensor1, empty_tensor);
    return empty_tensor;
}

}  // namespace ops

}  // namespace sail
