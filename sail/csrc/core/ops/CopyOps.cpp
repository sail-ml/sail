#include <iostream>

#include "CopyOps.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/Kernel.h"
#include "tensor_iterator.h"
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

void copy(Tensor& dest, const Tensor& source) {
    SAIL_CHECK(dest.get_shape().shape == source.get_shape().shape,
               "shapes dont match ", getVectorString(dest.get_shape().shape),
               " ", getVectorString(source.get_shape().shape));
    SAIL_CHECK(dest.get_dtype() == source.get_dtype());

    dispatch_all_types(dest.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        T* dst = static_cast<T*>(dest.get_data());
        T* src = static_cast<T*>(source.get_data());

        TensorShape d_shape = dest.get_shape();
        TensorShape s_shape = source.get_shape();

        MultiTensorIterator iter =
            MultiTensorIterator(d_shape).add_input(s_shape);
        int inner_loop_size = iter.inner_loop_size();
        int outer_steps = iter.out_loop_size();

        int z = 0;
        for (int i = 0; i < outer_steps; i++) {
            for (int j = 0; j < inner_loop_size; j += 1) {
                dst[iter.d_ptrs[0]] = src[iter.d_ptrs[1]];
                iter.advance_d_ptr(1);
                z += 1;
            }
            iter.backup_d_ptr();
            iter.next();
        }
    });
}

Tensor cast(Tensor& tensor1, Dtype dt) {
    TensorSize new_strides;
    long dt_size = GetDtypeSize(dt);
    // for (long s : tensor1.get_shape().shape) {
    //     new_strides.push_back(dt_size * s);
    // }
    Tensor empty_tensor =
        empty(tensor1.get_ndim(), dt, TensorShape(tensor1.get_shape().shape));

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
            T_out* nd = reinterpret_cast<T_out*>(d);
            ret = make_view((void*)nd, dt, t1.get_shape());
        });
    });
    return ret;
}

Tensor pad(Tensor& t1,
           std::vector<std::vector<long>>
               x) {  // const std::vector<std::tuple<long, long>> pads
    return sail::internal::pad_stub(t1, x);
}

/** end block **/

}  // namespace ops

}  // namespace sail
