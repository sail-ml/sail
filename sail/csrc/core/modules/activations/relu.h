#pragma once
#include "Tensor.h"
#include "modules/module.h"

namespace sail {
namespace modules {

class ReLU : public Module {
   public:
    explicit ReLU() = default;

    Tensor forward(Tensor& input);
};

}  // namespace modules
}  // namespace sail
