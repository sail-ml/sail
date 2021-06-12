#include "cross_entropy_loss.h"
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "modules/module.h"
#include "ops/ops.h"

namespace sail {
namespace loss {

Tensor SoftmaxCrossEntropyLoss::forward(Tensor& logits, Tensor& targets) {
    return ops::softmax_cross_entropy(logits, targets);
}

}  // namespace loss
}  // namespace sail