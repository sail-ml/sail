#include <algorithm>
#include "../Tensor.h"
#include "../TensorBody.h"
#include "../factories.h"
#include "../tensor_shape.h"
#include "copy.h"

namespace sail {

namespace ops {

Tensor broadcast_to(Tensor &tensor, TensorShape shape) {
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

    int indexer_2 = shape_base.ndim() - 1;
    for (int i = shape_new.ndim() - 1; i >= 0; i--) {
        if (indexer_2 < 0) {
            shape_new.strides[i] = 0;
        } else {
            if (shape_base.shape[indexer_2] != shape_new.shape[i] &&
                shape_base.shape[indexer_2] == 1) {
                shape_new.strides[i] = 0;
            } else if (shape_base.shape[indexer_2] == shape_new.shape[i]) {
                shape_new.strides[i] = shape_base.strides[indexer_2];
            } else {
                throw SailCError("shapes cannot be broadcasted together");
            }
        }
        indexer_2 -= 1;
    }

    shape_new.recompute();
    new_.set_shape(shape_new);

    return new_;
}

}  // namespace ops

}  // namespace sail