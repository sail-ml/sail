#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using reduction_fn_type = void (*)(const Tensor& t1, const int axis,
                                   Tensor& out);

DECLARE_DISPATCH(reduction_fn_type, sum_stub);
DECLARE_DISPATCH(reduction_fn_type, mean_stub);
DECLARE_DISPATCH(reduction_fn_type, max_stub);
DECLARE_DISPATCH(reduction_fn_type, min_stub);

}  // namespace internal

}  // namespace sail