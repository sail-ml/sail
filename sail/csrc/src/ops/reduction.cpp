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
    TensorSize old_shape = tensor1.get_shape().shape;
    TensorSize new_strides;
    TensorSize new_shape;
    long dt_size = GetDtypeSize(tensor1.get_dtype());
    int c = 0;
    for (long s : tensor1.get_shape().shape) {
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
    for (long sa : old_shape) {
        s *= sa;
        new_strides.push_back(s * dt_size);
        c += 1;
    }
    std::reverse(new_strides.begin(), new_strides.end());
    new_strides.push_back(dt_size);

    Tensor empty_tensor = empty(tensor1.get_ndim() - 1, tensor1.get_dtype(),
                                TensorShape(new_shape, new_strides));

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
Tensor mean(const Tensor& tensor1) {
    Tensor empty_tensor = empty_scalar(tensor1.get_dtype());
    int numel = empty_tensor.numel();
    int* ptr = &numel;
    SumTKernel().execute(tensor1, empty_tensor, -1);
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
