#include "relu.h"
#include <math.h> /* pow */
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "ops/ops.h"
#include "tensor_shape.h"

namespace sail {
namespace modules {

Tensor ReLU::forward(Tensor& input) { return sail::ops::relu(input); }

}  // namespace modules
}  // namespace sail
