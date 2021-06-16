#pragma once

#include <iostream>

#include "Tensor.h"
#include "factories.h"
#include "kernels/kernel.h"

namespace sail {
namespace ops {

Tensor ReLU(Tensor& input);

}  // namespace ops
}  // namespace sail
