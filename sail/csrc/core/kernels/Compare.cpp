
#include "kernels/Compare.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(clip_min_stub);
DEFINE_DISPATCH(clip_max_stub);
DEFINE_DISPATCH(clip_stub);

DEFINE_DISPATCH(not_equal_stub);
DEFINE_DISPATCH(equal_stub);
DEFINE_DISPATCH(lte_stub);
DEFINE_DISPATCH(gte_stub);
DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(lt_stub);
}  // namespace internal

}  // namespace sail