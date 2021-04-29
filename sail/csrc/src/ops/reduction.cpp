#include <iostream>

#include "../Tensor.h"
#include "../dtypes.h"
#include "../factories.h"
#include "../kernels/kernel.h"

namespace sail {
namespace ops {

Tensor sum(const Tensor& tensor1, int axis) {
    // Tensor empty_tensor = empty_scalar(tensor1.dtype);
    TensorSize new_strides;
    TensorSize new_shape;
    size_t dt_size = GetDtypeSize(tensor1.dtype);
    int c = 0;
    for (size_t s : tensor1.shape) {
        if (c != axis) {
            new_strides.push_back(dt_size * s);
        }
        new_shape.push_back(s);
        c += 1;
    }
    new_strides.pop_back();
    new_strides.push_back(dt_size);

    Tensor empty_tensor =
        empty(tensor1.ndim - 1, tensor1.dtype, new_strides, new_shape);

    SumTKernel().execute(tensor1, empty_tensor, axis);
    return empty_tensor;
}

Tensor sum(const Tensor& tensor1) {
    Tensor empty_tensor = empty_scalar(tensor1.dtype);

    SumTKernel().execute(tensor1, empty_tensor, -1);
    return empty_tensor;
}
Tensor mean(const Tensor& tensor1) {
    Tensor empty_tensor = empty_scalar(tensor1.dtype);
    int numel = empty_tensor.numel();
    int* ptr = &numel;
    MeanTKernel().execute(tensor1, empty_tensor);
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
