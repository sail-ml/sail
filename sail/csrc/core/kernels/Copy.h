#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using unary_fn_type = void (*)(const Tensor& t1, Tensor& out_tensor);
using pad_type = Tensor (*)(Tensor& t1, std::vector<std::vector<long>> pads);

DECLARE_DISPATCH(unary_fn_type, cast_stub);
DECLARE_DISPATCH(pad_type, pad_stub);

}  // namespace internal

}  // namespace sail