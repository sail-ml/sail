#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using reduction_fn_type2 = void (*)(const Tensor& t1, std::vector<long> axis,
                                    Tensor& out);
using mean_fn_type = void (*)(const Tensor& t1, std::vector<long> axis,
                              Tensor& out, long numel);

DECLARE_DISPATCH(reduction_fn_type2, sum_stub);
DECLARE_DISPATCH(mean_fn_type, mean_stub);
DECLARE_DISPATCH(reduction_fn_type2, max_stub);
DECLARE_DISPATCH(reduction_fn_type2, min_stub);

}  // namespace internal

}  // namespace sail