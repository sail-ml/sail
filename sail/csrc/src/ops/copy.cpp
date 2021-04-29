#include <iostream>

#include "../Tensor.h"
#include "../dtypes.h"
#include "../kernels/kernel.h"
#include "../types.h"
#include "copy.h"

#include "../factories.h"

namespace sail {

namespace ops {

Tensor copy(Tensor& tensor1) {
    Tensor empty_tensor;
    empty_tensor =
        empty(tensor1.ndim, tensor1.dtype, tensor1.strides, tensor1.shape);

    CopyTTKernel().execute(tensor1, empty_tensor);

    return empty_tensor;
}

Tensor cast(Tensor& tensor1, Dtype dt) {
    Tensor empty_tensor;
    TensorSize new_strides;
    long dt_size = GetDtypeSize(dt);
    for (long s : tensor1.shape) {
        new_strides.push_back(dt_size * s);
    }
    empty_tensor = empty(tensor1.ndim, dt, new_strides, tensor1.shape);

    CopyTTKernel().execute(tensor1, empty_tensor);

    return empty_tensor;
}

/** end block **/

}  // namespace ops

}  // namespace sail
