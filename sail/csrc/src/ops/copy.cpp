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
        empty(tensor1.get_ndim(), tensor1.get_dtype(), tensor1.get_shape());

    CopyTTKernel().execute(tensor1, empty_tensor);

    return empty_tensor;
}

Tensor cast(Tensor& tensor1, Dtype dt) {
    Tensor empty_tensor;
    TensorSize new_strides;
    long dt_size = GetDtypeSize(dt);
    // for (long s : tensor1.get_shape().shape) {
    //     new_strides.push_back(dt_size * s);
    // }
    empty_tensor = empty(tensor1.get_ndim(), dt, tensor1.get_shape());

    CopyTTKernel().execute(tensor1, empty_tensor);

    return empty_tensor;
}

Tensor view(Tensor& t1) {
    Tensor new_;
    // new_.set_data(t1.get_data());
    // new_.get_dtype() = t1.get_dtype();
    new_.fcn = t1.fcn;
    new_.grad = t1.grad;
    new_.requires_grad = t1.requires_grad;
    // new_.get_shape() = t1.get_shape();
    new_.has_grad = t1.has_grad;
    // new_.is_view()_base_shape = t1.get_shape();
    return new_;
}

/** end block **/

}  // namespace ops

}  // namespace sail
