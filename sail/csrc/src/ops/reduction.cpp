#include <algorithm>
#include <iostream>

#include "../Tensor.h"
#include "../dtypes.h"
#include "../factories.h"
#include "../kernels/kernel.h"

namespace sail {
namespace ops {

Tensor sum(const Tensor& tensor1, int axis) {
    // Tensor empty_tensor = empty_scalar(tensor1.dtype);
    TensorSize old_shape = tensor1.shape;
    TensorSize new_strides;
    TensorSize new_shape;
    size_t dt_size = GetDtypeSize(tensor1.dtype);
    int c = 0;
    for (size_t s : tensor1.shape) {
        if (c != axis) {
            new_shape.push_back(s);
        }
        c += 1;
    }

    old_shape.erase(old_shape.begin() + axis);
    std::reverse(old_shape.begin(), old_shape.end());
    old_shape.pop_back();
    long s = 1;
    c = 0;
    for (size_t sa : old_shape) {
        s *= sa;
        new_strides.push_back(s * dt_size);
        c += 1;
    }
    std::reverse(new_strides.begin(), new_strides.end());
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
