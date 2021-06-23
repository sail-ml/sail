#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using opt_fn = void (*)(Tensor& t1, Tensor& out_tensor,
                        const float learning_rate);

DECLARE_DISPATCH(opt_fn, sgd_stub);

}  // namespace internal

}  // namespace sail