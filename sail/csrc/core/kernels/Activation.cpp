#include "kernels/Activation.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(sigmoid_stub);
DEFINE_DISPATCH(sigmoid_backward_stub);
DEFINE_DISPATCH(softmax_stub);
DEFINE_DISPATCH(softmax_backward_partial_stub);
DEFINE_DISPATCH(softmax_mul_sum_stub);
}  // namespace internal

}  // namespace sail