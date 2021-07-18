#include <algorithm>
#include "Tensor.h"
#include "TensorBody.h"
#include "exception.h"
#include "factories.h"
#include "tensor_shape.h"

namespace sail {

namespace ops {

Tensor broadcast_to(const Tensor& input_tensor, TensorShape shape) {
    auto tensor = input_tensor;

    if (tensor.get_shape().shape == shape.shape) {
        return make_view(tensor);
    }
    if (tensor.is_view()) {
        tensor = clone(tensor);
    }
    Tensor new_ = make_view(tensor);

    TensorShape shape_base = tensor.get_shape();
    TensorShape shape_new = shape;

    if (tensor.is_scalar()) {
        LongVec t(shape_new.shape.size(), 0);
        shape_new.strides = t;
        shape_new.recompute();
        shape_new.is_single = true;

        new_.set_shape(shape_new);
        return new_;
    }

    auto& t2 = tensor;
    // while (t2.get_ndim() < shape.ndim()) {
    //     t2 = t2.expand_dims(0);
    // }

    TensorSize expand_shape, expand_strides;
    expand_shape = shape.shape;
    expand_strides = shape.strides;
    TensorSize tensor_strides = t2.get_shape().strides;
    TensorSize tensor_sizes = t2.get_shape().shape;
    TensorSize sizes = shape.shape;
    long ndim = shape.ndim();
    long tensor_dim = t2.get_ndim();

    for (long i = ndim - 1; i >= 0; --i) {
        long offset = ndim - 1 - i;
        long dim = tensor_dim - 1 - offset;
        long size = (dim >= 0) ? tensor_sizes[dim] : 1;
        long stride = (dim >= 0) ? tensor_strides[dim]
                                 : expand_shape[i + 1] * expand_strides[i + 1];
        long targetSize = sizes[i];
        if (targetSize == -1) {
            SAIL_CHECK(
                dim >= 0, "The expanded size of the tensor (", targetSize,
                ") isn't allowed in a leading, non-existing dimension ", i);
            targetSize = size;
        }
        if (size != targetSize) {
            SAIL_CHECK(size == 1, "Tensor shapes must match at dimension ", i,
                       ". Target shape: ", getVectorString(sizes),
                       ". Input shape: ", getVectorString(tensor_sizes));
            size = targetSize;
            stride = 0;
        }
        expand_shape[i] = size;
        expand_strides[i] = stride;
    }

    shape_new = TensorShape(expand_shape, expand_strides);

    shape_new.recompute();
    // shape_new.contiguous = false;
    new_.set_shape(shape_new);

    return new_;
}
TensorShape broadcast_to_shape_only(const TensorShape shape_in,
                                    TensorShape shape) {
    TensorShape shape_base = shape_in;
    TensorShape shape_new = shape;

    // while (t2.get_ndim() < shape.ndim()) {
    //     t2 = t2.expand_dims(0);
    // }

    TensorSize expand_shape, expand_strides;
    expand_shape = shape.shape;
    expand_strides = shape.strides;
    TensorSize tensor_strides = shape_in.strides;
    TensorSize tensor_sizes = shape_in.shape;
    TensorSize sizes = shape.shape;
    long ndim = shape.ndim();
    long tensor_dim = shape_in.ndim();

    for (long i = ndim - 1; i >= 0; --i) {
        long offset = ndim - 1 - i;
        long dim = tensor_dim - 1 - offset;
        long size = (dim >= 0) ? tensor_sizes[dim] : 1;
        long stride = (dim >= 0) ? tensor_strides[dim]
                                 : expand_shape[i + 1] * expand_strides[i + 1];
        long targetSize = sizes[i];
        if (targetSize == -1) {
            SAIL_CHECK(
                dim >= 0, "The expanded size of the tensor (", targetSize,
                ") isn't allowed in a leading, non-existing dimension ", i);
            targetSize = size;
        }
        if (size != targetSize) {
            SAIL_CHECK(size == 1, "Tensor shapes must match at dimension ", i,
                       ". Target shape: ", getVectorString(sizes),
                       ". Input shape: ", getVectorString(tensor_sizes));
            size = targetSize;
            stride = 0;
        }
        expand_shape[i] = size;
        expand_strides[i] = stride;
    }

    shape_new = TensorShape(expand_shape, expand_strides);

    shape_new.recompute();
    // shape_new.contiguous = false;

    return shape_new;
}

}  // namespace ops

}  // namespace sail