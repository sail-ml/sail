#pragma once

#include <iostream>

#include "../Tensor.h"
#include "../kernels/kernel.h"

namespace sail {

namespace ops {


Tensor add(const Tensor& tensor1, const Tensor& tensor2);
Tensor subtract(const Tensor& tensor1, const Tensor& tensor2);
Tensor multiply(const Tensor& tensor1, const Tensor& tensor2);
Tensor divide(const Tensor& tensor1, const Tensor& tensor2);

} // end ops

} // end sail
