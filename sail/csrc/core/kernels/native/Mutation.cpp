#include "kernels/Mutation.h"
#include "Tensor.h"
#include "dtypes.h"
#include "exception.h"
#include "factories.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "tensor_shape.h"
#include "utils.h"

namespace sail {

namespace internal {

namespace {

Tensor cat_kernel(std::vector<Tensor> tensors, const int axis, const int cat) {
    SAIL_CHECK(tensors.size() > 1, "Must pass more than one tensor");

    int ndim = tensors[0].get_ndim();

    TensorShape check = tensors[0].get_shape();
    std::vector<long> combined = check.shape;
    Dtype dt = tensors[0].get_dtype();
    combined[axis] = 0;

    for (auto i : sail::irange(0, (int)tensors.size())) {
        if (tensors[i].get_ndim() != ndim) {
            THROW_ERROR_DETAILED(DimensionError,
                                 "Number of dimensions must match");
        }
        TensorShape s = tensors[i].get_shape();
        long save = s[axis];
        long o_save = combined[axis];
        s.shape[axis] = 0;
        combined[axis] = 0;
        SAIL_CHECK(s.shape == combined,
                   "Shapes do not match when ignoring index ", axis);
        if (cat == 1) {
            combined[axis] = save + o_save;
        } else {
            combined[axis] = 1;
        }

        if (tensors[i].is_view()) {
            tensors[i] = clone(tensors[i]);
        }
        dt = promote_dtype(dt, tensors[i].get_dtype());
    }

    Tensor out = empty(0, dt, TensorShape(combined));

    dispatch_all_types(out.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        long outer = out.numel() /
                     (out.get_shape()[axis] * out.get_shape().strides[axis]);
        T* result_ptr = (T*)(out.get_data());

        for (int i = 0; i < outer; i++) {
            for (const auto& t : tensors) {
                // MultiTensorIterator inner_iter =
                //     MultiTensorIterator(t.get_shape());
                int64_t local_inner =
                    t.get_shape()[axis] *
                    t.get_shape()
                        .strides[axis];  // inner_iter.inner_loop_size();
                T* input_ptr = (T*)(t.get_data()) + i * local_inner;
                int64_t d = 0;
                for (; d < local_inner; d++) {
                    result_ptr[d] = input_ptr[d];
                }
                result_ptr += local_inner;
            }
        }
    });
    return out;
}

Tensor stack_kernel(std::vector<Tensor> tensors, const int axis) {
    SAIL_CHECK(tensors.size() > 1, "Must pass more than one tensor");

    TensorShape check = tensors[0].get_shape();
    std::vector<long> combined = check.shape;
    Dtype dt = tensors[0].get_dtype();
    for (auto i : sail::irange(0, (int)tensors.size())) {
        if (tensors[i].is_view()) {
            tensors[i] = clone(tensors[i]);
        }
        dt = promote_dtype(dt, tensors[i].get_dtype());
    }
    int new_axis = axis;
    if (axis < 0) {
        new_axis = new_axis + combined.size() + 1;
    }
    combined.insert(combined.begin() + new_axis, tensors.size());

    TensorShape x = TensorShape(combined);
    Tensor out = empty(0, dt, x);

    dispatch_all_types(out.get_dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        long outer = out.numel() /
                     (out.get_shape()[axis] * out.get_shape().strides[axis]);
        T* result_ptr = (T*)(out.get_data());

        for (int i = 0; i < outer; i++) {
            for (const auto& t : tensors) {
                TensorShape x = t.get_shape();
                x.insert_one(axis);
                int64_t local_inner =
                    x[axis] * x.strides[axis];  // inner_iter.inner_loop_size();
                T* input_ptr = (T*)(t.get_data()) + i * local_inner;
                int64_t d = 0;
                for (; d < local_inner; d++) {
                    result_ptr[d] = input_ptr[d];
                }
                result_ptr += local_inner;
            }
        }
    });
    return out;
}

}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(cat_stub, &cat_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(stack_stub, &stack_kernel);

}  // namespace internal

}  // namespace sail