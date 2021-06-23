#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using binary_fn_type = void (*)(const Tensor& t1, const Tensor& t2, Tensor& out,
                                bool broadcast);
using unary_fn_type = void (*)(const Tensor& t1, Tensor& out);

DECLARE_DISPATCH(binary_fn_type, power_stub);
DECLARE_DISPATCH(unary_fn_type, exp_stub);
DECLARE_DISPATCH(unary_fn_type, log_stub);

}  // namespace internal

}  // namespace sail