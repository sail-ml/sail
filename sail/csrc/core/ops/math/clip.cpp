#include "Tensor.h"
#include "autograd/autograd.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/Kernel.h"

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor clip(Tensor& tensor1, double min) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::ClipMinOnly(min))
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    empty_tensor = empty(0, tensor1.get_dtype(), tensor1.get_shape());
    sail::internal::clip_min_stub(tensor1, min, empty_tensor);
    return empty_tensor;
}

Tensor clip(Tensor& tensor1, double min, double max) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Clip(min, max))
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    empty_tensor = empty_like(tensor1);
    sail::internal::clip_stub(tensor1, min, max, empty_tensor);
    return empty_tensor;
}
}  // namespace ops

}  // namespace sail
