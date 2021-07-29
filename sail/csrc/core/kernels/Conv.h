#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using conv2d_fn = std::vector<Tensor> (*)(Tensor& input, Tensor& kernel,
                                          std::vector<long> stride,
                                          std::string padding_mode);

DECLARE_DISPATCH(conv2d_fn, conv2d_stub);

}  // namespace internal

}  // namespace sail