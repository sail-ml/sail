#pragma once

#include <iostream>
#include <sstream>

#include "Tensor.h"
#include "factories.h"
#include "kernels/Kernel.h"

namespace sail {
namespace ops {

std::string tensor_repr(Tensor& array);

}  // namespace ops
}  // namespace sail
