#include "kernels/Reduction.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(mean_stub);
DEFINE_DISPATCH(max_stub);
DEFINE_DISPATCH(min_stub);
}  // namespace internal

}  // namespace sail