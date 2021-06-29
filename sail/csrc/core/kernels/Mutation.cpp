#include "kernels/Mutation.h"
#include <iostream>
#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(cat_stub);
DEFINE_DISPATCH(stack_stub);

}  // namespace internal

}  // namespace sail