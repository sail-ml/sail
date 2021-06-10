#pragma once
#include "../Tensor.h"
#include "../dtypes.h"
#include "../factories.h"
#include "../modules/module.h"
#include "../tensor_shape.h"
#include "optimizers.h"
namespace sail {
namespace optimizers {

using TensorVector = std::vector<Tensor>;

class SGD : public Optimizer {
   public:
    float learning_rate;
    SGD(float _learning_rate) { learning_rate = _learning_rate; };
    void update();
};
}  // namespace optimizers
}  // namespace sail
