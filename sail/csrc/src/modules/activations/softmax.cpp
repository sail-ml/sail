#include "softmax.h"
#include <math.h> /* pow */
#include "../../Tensor.h"
#include "../../dtypes.h"
#include "../../factories.h"
#include "../../ops/ops.h"
#include "../../tensor_shape.h"

namespace sail {
namespace modules {

Tensor Softmax::forward(Tensor& input) { return ops::softmax(input, axis); }

}  // namespace modules
}  // namespace sail
