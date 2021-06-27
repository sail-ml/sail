#include <algorithm>
#include "Tensor.h"
#include "TensorBody.h"
#include "factories.h"
#include "tensor_shape.h"

namespace sail {

namespace ops {

Tensor broadcast_to(const Tensor &tensor, TensorShape shape) {
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

    Tensor t2 = tensor;
    // while (t2.get_ndim() < shape.ndim()) {
    //     t2 = t2.expand_dims(0);
    // }

    TensorSize expand_shape, expand_strides;
    expand_shape = shape.shape;
    expand_strides = shape.strides;
    TensorSize tensor_strides = t2.get_shape().strides;
    TensorSize tensor_sizes = t2.get_shape().shape;
    TensorSize sizes = shape.shape;
    int ndim = shape.ndim();
    int tensor_dim = t2.get_ndim();

    for (int64_t i = ndim - 1; i >= 0; --i) {
        int64_t offset = ndim - 1 - i;
        int64_t dim = tensor_dim - 1 - offset;
        int64_t size = (dim >= 0) ? tensor_sizes[dim] : 1;
        int64_t stride = (dim >= 0)
                             ? tensor_strides[dim]
                             : expand_shape[i + 1] * expand_strides[i + 1];
        int64_t targetSize = sizes[i];
        if (targetSize == -1) {
            targetSize = size;
        }
        if (size != targetSize) {
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

}  // namespace ops

}  // namespace sail