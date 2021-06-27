#pragma once

#include <iostream>
#include <sstream>
#include "Tensor.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

DECLARE_DISPATCH(void (*)(const Tensor& t1, std::ostream& os), repr_stub);

}  // namespace internal

}  // namespace sail