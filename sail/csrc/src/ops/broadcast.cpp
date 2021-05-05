#include "../Tensor.h"
#include "../tensor_shape.h"

namespace sail {

namespace ops {

Tensor broadcast_to(Tensor& tensor, TensorShape shape) {
    tensor.old_shape = tensor.shape_details;
    tensor.shape_details = shape;
    tensor.broadcasted = true;
    return tensor;
}

}  // namespace ops

}  // namespace sail