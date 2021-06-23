
#pragma once

#include <iostream>

#include "Tensor.h"
#include "factories.h"
#include "kernels/Kernel.h"

namespace sail {
namespace ops {

Tensor mean_squared_error(Tensor& logits, Tensor& targets);

}  // namespace ops
}  // namespace sail
