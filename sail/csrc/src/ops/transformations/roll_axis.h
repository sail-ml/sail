#pragma once

#include <algorithm>
#include <iostream>

#include "../../Tensor.h"
#include "../../error.h"
#include "../../factories.h"
#include "../../types.h"

namespace sail {

namespace ops {

Tensor roll_axis(const Tensor& tensor1, const int axis, const int position);

}  // namespace ops

}  // namespace sail
