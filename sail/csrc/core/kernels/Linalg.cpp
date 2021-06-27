#include "kernels/Linalg.h"
#include <iostream>
#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DEFINE_DISPATCH(matmul_stub);

}  // namespace internal

}  // namespace sail