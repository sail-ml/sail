#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using cat_fn = Tensor (*)(std::vector<Tensor> tensors, const int axis,
                          const int cat);

DECLARE_DISPATCH(cat_fn, cat_stub);
using stack_fn = Tensor (*)(std::vector<Tensor> tensors, const int axis);

DECLARE_DISPATCH(stack_fn, stack_stub);

}  // namespace internal

}  // namespace sail