#include "mean_squared_error.h"
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "modules/module.h"
#include "ops/ops.h"

namespace sail {
namespace loss {

Tensor MeanSquaredError::forward(Tensor& logits, Tensor& targets) {
    return ops::mean_squared_error(logits, targets);
}

}  // namespace loss
}  // namespace sail