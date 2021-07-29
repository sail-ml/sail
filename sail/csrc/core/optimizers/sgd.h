// allow-impl-in-header

#pragma once
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "modules/module.h"
#include "optimizers.h"
#include "tensor_shape.h"

#define DISABLE_GRAD(a, b)                     \
    {                                          \
        a.was_requires_grad = a.requires_grad; \
        a.requires_grad = false;               \
        b.was_requires_grad = b.requires_grad; \
        b.requires_grad = false;               \
    }
#define ENABLE_GRAD(a, b)                      \
    {                                          \
        a.requires_grad = a.was_requires_grad; \
        b.requires_grad = b.was_requires_grad; \
    }

namespace sail {
namespace optimizers {

using TensorVector = std::vector<Tensor>;

class SGD : public Optimizer {
   public:
    float learning_rate;
    SGD(float _learning_rate) { learning_rate = _learning_rate; };
    void update() override;
};
}  // namespace optimizers
}  // namespace sail
