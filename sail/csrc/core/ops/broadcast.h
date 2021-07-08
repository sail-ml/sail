
#include "Tensor.h"
#include "tensor_shape.h"

namespace sail {

namespace ops {

Tensor broadcast_to(const Tensor& tensor, TensorShape shape);
TensorShape broadcast_to_shape_only(const TensorShape shape_in,
                                    TensorShape shape);
}  // namespace ops

}  // namespace sail