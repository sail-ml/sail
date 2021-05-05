#pragma once

#include <iostream>
#include <sstream>

#include "../Tensor.h"
#include "../factories.h"
#include "../kernels/kernel.h"

namespace sail {
namespace ops {

inline std::ostream& operator<<(std::ostream& os, Tensor& tensor) {
    // TODO(hvy): We need to determine the output specification of this
    // function, whether or not to align with Python repr specification, and
    // also whether this functionality should be defined in C++ layer or Python
    // layer.
    // TODO(hvy): Consider using a static dimensionality.
    ReprKernel().execute(tensor, os);
    return os;
}

inline std::string tensor_repr(Tensor& array) {
    std::ostringstream os;
    os << array;
    return os.str();
}

}  // namespace ops
}  // namespace sail
