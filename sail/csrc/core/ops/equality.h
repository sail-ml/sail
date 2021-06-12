#pragma once

#include <iostream>

#include "Tensor.h"
#include "factories.h"
#include "kernels/kernel.h"

namespace sail {
namespace ops {

Tensor elementwise_equal(const Tensor& tensor1, const Tensor& tensor2);

}  // namespace ops
}  // namespace sail
