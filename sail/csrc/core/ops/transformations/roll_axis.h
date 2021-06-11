#pragma once

#include <algorithm>
#include <iostream>

#include "Tensor.h"
#include "error.h"
#include "factories.h"
#include "types.h"

namespace sail {

namespace ops {

Tensor rollaxis(const Tensor& tensor1, const int axis, const int position = 0);

}  // namespace ops

}  // namespace sail
