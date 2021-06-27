#include "kernels/Binary.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(subtract_stub);
DEFINE_DISPATCH(divide_stub);
DEFINE_DISPATCH(multiply_stub);
}  // namespace internal

}  // namespace sail