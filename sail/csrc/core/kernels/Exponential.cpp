#include "kernels/Exponential.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(power_stub);
DEFINE_DISPATCH(exp_stub);
DEFINE_DISPATCH(log_stub);
}  // namespace internal

}  // namespace sail