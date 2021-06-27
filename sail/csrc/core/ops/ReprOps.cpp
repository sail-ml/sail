#pragma once

#include <iostream>
#include <sstream>

#include "ReprOps.h"
#include "Tensor.h"
#include "kernels/Kernel.h"

namespace sail {
namespace ops {

std::string tensor_repr(Tensor& array) {
    std::ostringstream os;
    os << array;
    return os.str();
}

}  // namespace ops
}  // namespace sail
