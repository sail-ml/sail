#include "../Tensor.h"
#include "../tensor_shape.h"

namespace sail {

namespace ops {

Tensor broadcast_to(Tensor& tensor, TensorShape shape) {
    tensor.shape_details = shape;
    return tensor;
}

}  // namespace ops

}  // namespace sail