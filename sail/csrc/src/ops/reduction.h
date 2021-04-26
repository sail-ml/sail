#pragma once

#include <iostream>

#include "../Tensor.h"
#include "../factories.h"
#include "../kernels/kernel.h"

namespace sail {
namespace ops {

Tensor sum(const Tensor& tensor1);
Tensor mean(const Tensor& tensor1);

}
} // end sail
