#include "sigmoid.h"
#include <cmath> /* pow */
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {
namespace modules {

Tensor Sigmoid::forward(Tensor& input) { return ops::sigmoid(input); }

}  // namespace modules
}  // namespace sail
