#pragma once

#include <iostream>
#include <sstream>

#include "Tensor.h"
#include "factories.h"
#include "kernels/Kernel.h"

namespace sail {
namespace ops {

inline std::string tensor_repr(Tensor& array) {
    std::ostringstream os;
    os << array;
    return os.str();
}

}  // namespace ops
}  // namespace sail
