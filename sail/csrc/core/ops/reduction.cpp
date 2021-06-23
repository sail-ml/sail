#include <algorithm>
#include <iostream>

#include "Tensor.h"
#include "autograd/reduction_function.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/Kernel.h"

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

Tensor sum(const Tensor& tensor1, int axis = NULLDIM, bool keepdims = false) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Sum(axis, keepdims))
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    // Tensor empty_tensor = empty_scalar(tensor1.get_dtype());
    TensorShape new_shape;
    if (axis == NULLDIM) {
        new_shape = TensorShape({1});
    } else {
        new_shape = TensorShape(tensor1.get_shape());
        new_shape.remove(axis);
    }

    empty_tensor = zeros(TensorShape(new_shape), tensor1.get_dtype());

    sail::internal::sum_stub(tensor1, axis, empty_tensor);
    if (keepdims) {
        empty_tensor = empty_tensor._expand_dims_inplace(axis);
    }
    return empty_tensor;
}

Tensor mean(const Tensor& tensor1, int axis = NULLDIM, bool keepdims = false) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Mean(axis, keepdims))
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    // Tensor empty_tensor = empty_scalar(tensor1.get_dtype());
    TensorShape new_shape;
    if (axis == NULLDIM) {
        new_shape = TensorShape({1});
    } else {
        new_shape = TensorShape(tensor1.get_shape());
        new_shape.remove(axis);
    }

    empty_tensor = zeros(TensorShape(new_shape), tensor1.get_dtype());

    sail::internal::mean_stub(tensor1, axis, empty_tensor);
    if (keepdims) {
        empty_tensor = empty_tensor._expand_dims_inplace(axis);
    }
    return empty_tensor;
}

Tensor max(const Tensor& tensor1, int axis = NULLDIM, bool keepdims = false) {
    Tensor empty_tensor;
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Max(axis, keepdims))
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }

    TensorShape new_shape;
    if (axis == NULLDIM) {
        new_shape = TensorShape({1});
    } else {
        new_shape = TensorShape(tensor1.get_shape());
        new_shape.remove(axis);
    }

    empty_tensor = zeros(TensorShape(new_shape), tensor1.get_dtype());

    sail::internal::max_stub(tensor1, axis, empty_tensor);
    if (keepdims) {
        empty_tensor = empty_tensor._expand_dims_inplace(axis);
    }
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
