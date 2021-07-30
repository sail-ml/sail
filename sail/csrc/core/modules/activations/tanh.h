#pragma once
#include "Tensor.h"
#include "modules/module.h"

namespace sail {
namespace modules {

class Tanh : public Module {
   public:
    explicit Tanh() = default;

    Tensor forward(Tensor& input);
};

}  // namespace modules
}  // namespace sail
