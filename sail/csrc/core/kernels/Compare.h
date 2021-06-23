#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using clip_single = void (*)(const Tensor& t1, const double min, Tensor& out);
using clip_both = void (*)(const Tensor& t1, const double min, const double max,
                           Tensor& out);

DECLARE_DISPATCH(clip_single, clip_min_stub);
DECLARE_DISPATCH(clip_single, clip_max_stub);
DECLARE_DISPATCH(clip_both, clip_stub);

using elementwise_equal = void (*)(const Tensor& t1, const Tensor& t2,
                                   const Tensor& out_tensor, bool broadcast);
DECLARE_DISPATCH(elementwise_equal, equal_stub);
DECLARE_DISPATCH(elementwise_equal, lte_stub);
DECLARE_DISPATCH(elementwise_equal, gte_stub);
DECLARE_DISPATCH(elementwise_equal, gt_stub);
DECLARE_DISPATCH(elementwise_equal, lt_stub);

}  // namespace internal

}  // namespace sail