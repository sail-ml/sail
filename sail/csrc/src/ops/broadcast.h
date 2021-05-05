
#include "../Tensor.h"
#include "../tensor_shape.h"

namespace sail {

namespace ops {

Tensor broadcast_to(Tensor& tensor, TensorShape shape);
}

}  // namespace sail