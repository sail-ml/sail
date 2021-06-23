#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using binary_fn_type = void (*)(const Tensor& t1, const Tensor& t2, Tensor& out,
                                bool broadcast);

DECLARE_DISPATCH(binary_fn_type, add_stub);
DECLARE_DISPATCH(binary_fn_type, subtract_stub);
DECLARE_DISPATCH(binary_fn_type, multiply_stub);
DECLARE_DISPATCH(binary_fn_type, divide_stub);

}  // namespace internal

}  // namespace sail