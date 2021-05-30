#pragma once
#include "../../Tensor.h"
#include "../module.h"

namespace sail {
namespace modules {

class Sigmoid : public Module {
   public:

    explicit Sigmoid(){};

    Tensor forward(Tensor& input);
};

}  // namespace modules
}  // namespace sail
