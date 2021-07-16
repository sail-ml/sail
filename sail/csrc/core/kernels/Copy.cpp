#include "kernels/Copy.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(copy_stub);
DEFINE_DISPATCH(cast_stub);
DEFINE_DISPATCH(pad_stub);

}  // namespace internal

}  // namespace sail