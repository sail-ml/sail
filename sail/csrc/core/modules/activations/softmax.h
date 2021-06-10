#pragma once
#include "../../Tensor.h"
#include "../module.h"

namespace sail {
namespace modules {

class Softmax : public Module {
   public:
    int axis;
    Softmax(int _axis = 1) : axis(_axis){};

    Tensor forward(Tensor& input);
};

}  // namespace modules
}  // namespace sail
