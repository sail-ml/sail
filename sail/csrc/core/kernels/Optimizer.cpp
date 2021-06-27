#include "kernels/Optimizer.h"
#include <iostream>
#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(sgd_stub);

}  // namespace internal

}  // namespace sail