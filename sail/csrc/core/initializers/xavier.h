#pragma once
#include "Tensor.h"

namespace sail {

namespace initializers {
Tensor xavier_uniform(Tensor input, double gain = 1.0);
Tensor xavier_normal(Tensor input, double gain = 1.0);
}  // namespace initializers

}  // namespace sail