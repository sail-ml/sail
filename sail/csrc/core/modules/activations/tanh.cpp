#include "tanh.h"
#include <cmath> /* pow */
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {
namespace modules {

Tensor Tanh::forward(Tensor& input) { return sail::ops::tanh(input); }

}  // namespace modules
}  // namespace sail
