
#include "Tensor.h"
#include "tensor_shape.h"

namespace sail {

namespace ops {

Tensor broadcast_to(const Tensor& tensor, TensorShape shape);
}  // namespace ops

}  // namespace sail