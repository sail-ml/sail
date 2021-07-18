#pragma once
#include "Tensor.h"
#include "modules/module.h"

namespace sail {
namespace loss {

class SoftmaxCrossEntropyLoss : public sail::modules::Module {
   public:
    explicit SoftmaxCrossEntropyLoss() = default;

    Tensor forward(Tensor& logits, Tensor& targets);
};

}  // namespace loss
}  // namespace sail