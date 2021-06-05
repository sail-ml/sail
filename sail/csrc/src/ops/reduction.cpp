#include <algorithm>
#include <iostream>

#include "../Tensor.h"
#include "../autograd/reduction_function.h"
#include "../dtypes.h"
#include "../factories.h"
#include "../kernels/kernel.h"

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor sum(const Tensor& tensor1, int axis) {
    // Tensor empty_tensor = empty_scalar(tensor1.get_dtype());
    TensorShape new_shape = TensorShape(tensor1.get_shape());
    new_shape.remove(axis);

    Tensor empty_tensor = empty(0, tensor1.get_dtype(), TensorShape(new_shape));

    SumTKernel().execute(tensor1, empty_tensor, axis);
    return empty_tensor;
}

Tensor sum(const Tensor& tensor1) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Sum())
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    empty_tensor = zero_scalar(tensor1.get_dtype());

    SumTKernel().execute(tensor1, empty_tensor, -1);
    return empty_tensor;
}

Tensor max(const Tensor& tensor1, int axis) {
    // Tensor empty_tensor = empty_scalar(tensor1.get_dtype());
    TensorShape new_shape = TensorShape(tensor1.get_shape());
    new_shape.remove(axis);

    Tensor empty_tensor = empty(0, tensor1.get_dtype(), TensorShape(new_shape));

    MaxKernel().execute(tensor1, empty_tensor, axis);
    return empty_tensor;
}

Tensor max(const Tensor& tensor1) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Sum())
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    empty_tensor = zero_scalar(tensor1.get_dtype());

    MaxKernel().execute(tensor1, empty_tensor, -1);
    return empty_tensor;
}
Tensor mean(const Tensor& tensor1) {
    Tensor empty_tensor = empty_scalar(tensor1.get_dtype());
    int numel = empty_tensor.numel();
    int* ptr = &numel;
    SumTKernel().execute(tensor1, empty_tensor, -1);
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
