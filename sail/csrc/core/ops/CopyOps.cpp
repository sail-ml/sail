#include <iostream>

#include "CopyOps.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/Kernel.h"
#include "types.h"

#include "factories.h"

namespace sail {

namespace ops {

Tensor copy(Tensor& tensor1) {
    Tensor empty_tensor;
    empty_tensor =
        empty(tensor1.get_ndim(), tensor1.get_dtype(), tensor1.get_shape());

    sail::internal::cast_stub(tensor1, empty_tensor);  // change to basic copy

    return empty_tensor;
}

Tensor cast(Tensor& tensor1, Dtype dt) {
    TensorSize new_strides;
    long dt_size = GetDtypeSize(dt);
    // for (long s : tensor1.get_shape().shape) {
    //     new_strides.push_back(dt_size * s);
    // }
    Tensor empty_tensor = empty(tensor1.get_ndim(), dt, tensor1.get_shape());

    sail::internal::cast_stub(tensor1, empty_tensor);

    return empty_tensor;
}

Tensor view(Tensor& t1) {
    Tensor new_;
    // new_.set_data(t1.get_data());
    // new_.get_dtype() = t1.get_dtype();
    new_.fcn = t1.fcn;
    new_.requires_grad = t1.requires_grad;
    // new_.get_shape() = t1.get_shape();
    // new_.is_view()_base_shape = t1.get_shape();
    return new_;
}

Tensor internal_fast_cast(Tensor& t1, Dtype dt) {
    Tensor ret;
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        dispatch_all_types(dt, [&](auto pt2) {
            using T_in = typename decltype(pt)::type;
            using T_out = typename decltype(pt2)::type;

            T_in* d = static_cast<T_in*>(t1.get_data());
            std::cout << d[0] << std::endl;
            T_out* nd = reinterpret_cast<T_out*>(d);
            std::cout << nd[0] << std::endl;
            ret = make_view((void*)nd, dt, t1.get_shape());
        });
    });
    return ret;
}

/** end block **/

}  // namespace ops

}  // namespace sail
