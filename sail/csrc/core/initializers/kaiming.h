#pragma once
#include "Tensor.h"

namespace sail {

namespace initializers {
Tensor kaiming_uniform(Tensor input, std::string mode = "fan_in",
                       std::string nonlin = "leaky_relu");
Tensor kaiming_normal(Tensor input, std::string mode = "fan_in",
                      std::string nonlin = "leaky_relu");
}  // namespace initializers

}  // namespace sail