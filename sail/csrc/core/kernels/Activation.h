#pragma once

#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

using unary_fn_type = void (*)(const Tensor& t1, Tensor& out);
using softmax_fn_type = void (*)(Tensor& t1, const int axis, Tensor& out);

using softmax_back = void (*)(Tensor& y, Tensor& targets, Tensor& out_tensor);

DECLARE_DISPATCH(unary_fn_type, tanh_stub);
DECLARE_DISPATCH(softmax_fn_type, softmax_stub);
DECLARE_DISPATCH(unary_fn_type, sigmoid_stub);
DECLARE_DISPATCH(unary_fn_type, sigmoid_backward_stub);
DECLARE_DISPATCH(softmax_back, softmax_backward_partial_stub);
DECLARE_DISPATCH(softmax_back, softmax_mul_sum_stub);

}  // namespace internal

}  // namespace sail