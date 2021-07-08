#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using im2col_fn = Tensor (*)(Tensor& input, std::vector<long> kernel_size,
                             std::vector<long> stride, std::vector<long> pads);

DECLARE_DISPATCH(im2col_fn, im2col_stub);

}  // namespace internal

}  // namespace sail