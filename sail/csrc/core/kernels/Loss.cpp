#include "kernels/Loss.h"
#include <iostream>
#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(mse_stub);

}  // namespace internal

}  // namespace sail