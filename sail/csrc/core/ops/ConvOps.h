#pragma once

#include <iostream>

#include "Tensor.h"
#include "dtypes.h"
#include "kernels/Kernel.h"

namespace sail {

namespace ops {

Tensor conv2d(Tensor& x, Tensor& w, std::vector<long> stride,
              std::string padding_mode = "same");
}  // namespace ops

}  // namespace sail
