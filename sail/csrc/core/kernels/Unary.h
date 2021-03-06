#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using unary_fn_type = void (*)(const Tensor& t1, Tensor& out);

DECLARE_DISPATCH(unary_fn_type, negate_stub);

}  // namespace internal

}  // namespace sail