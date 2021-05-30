#include <algorithm>
#include "../Tensor.h"
#include "../TensorBody.h"
#include "../factories.h"
#include "../tensor_shape.h"
#include "copy.h"

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

        new_.set_shape(shape_new);
        return new_;
    }

    int i1 = shape_new.ndim() - 1;
    int i2 = shape_base.ndim() - 1;

    while (i1 >= 0) {
        if (i2 < 0) {
            shape_new.strides[i1] = 0;
        } else {
            if (shape_base.shape[i2] != shape_new.shape[i1] &&
                shape_base.shape[i2] == 1) {
                shape_new.strides[i1] = 0;
            } else if (shape_base.shape[i2] == shape_new.shape[i1]) {
                shape_new.strides[i1] = shape_base.strides[i2];
            } else {
                throw SailCError("shapes cannot be broadcasted together");
            }
        }
        i1--;
        i2--;
    }

    shape_new.recompute();
    new_.set_shape(shape_new);

    return new_;
}

}  // namespace ops

}  // namespace sail