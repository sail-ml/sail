#include <algorithm>
#include <iostream>

#include "ReductionOps.h"
#include "Tensor.h"
#include "autograd/autograd.h"
#include "constants.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/Kernel.h"
#include "utils.h"

namespace sail {
namespace ops {
using TensorVector = std::vector<Tensor>;

TensorShape shape_process(const Tensor& tensor1, std::vector<long> axis) {
    TensorShape new_shape;
    std::vector<long> sh;
    if (axis[0] == NULLDIM) {
        new_shape = TensorShape({1});
    } else {
        std::vector<long> sh;
        int i = 0;
        for (long s : tensor1.get_shape().shape) {
            if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
                sh.push_back(s);
            }
            i += 1;
        }
        new_shape = TensorShape(sh);
    }
    return new_shape;
}

std::vector<long> process_axes(const long ndim, std::vector<long> axes) {
    // for (int i = 0; i < axes.size(); i++) {
    for (const auto i : sail::irange(0, (int)axes.size())) {
        if (axes[i] < 0) {
            axes[i] = axes[i] + ndim;
        }
    }
    return axes;
}

Tensor sum(const Tensor& tensor1, int axis, bool keepdims) {
    std::vector<long> axes = {axis};
    return sum(tensor1, axes, keepdims);
}
Tensor sum(const Tensor& tensor1, std::vector<long> axis, bool keepdims) {
    Tensor empty_tensor;
    TensorShape input_shape = tensor1.get_shape();
    axis = process_axes(tensor1.get_ndim(), axis);
    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Sum(axis, keepdims))
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    // Tensor empty_tensor = empty_scalar(tensor1.get_dtype());

    auto new_shape = shape_process(tensor1, axis);
    empty_tensor = zeros(TensorShape(new_shape), tensor1.get_dtype());

    sail::internal::sum_stub(tensor1, axis, empty_tensor);
    if (keepdims) {
        if (axis[0] == NULLDIM) {
            empty_tensor._inplace_reshape(input_shape);
        } else {
            for (int a : axis) {
                empty_tensor = empty_tensor._expand_dims_inplace(a);
            }
        }
    }
    return empty_tensor;
}

Tensor mean(const Tensor& tensor1, int axis, bool keepdims) {
    std::vector<long> axes = {axis};
    return mean(tensor1, axes, keepdims);
}
Tensor mean(const Tensor& tensor1, std::vector<long> axis, bool keepdims) {
    Tensor empty_tensor;
    TensorShape input_shape = tensor1.get_shape();
    axis = process_axes(tensor1.get_ndim(), axis);

    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Mean(axis, keepdims))
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    // Tensor empty_tensor = empty_scalar(tensor1.get_dtype());

    auto new_shape = shape_process(tensor1, axis);
    empty_tensor = zeros(TensorShape(new_shape), tensor1.get_dtype());

    sail::internal::mean_stub(tensor1, axis, empty_tensor);
    if (keepdims) {
        if (axis[0] == NULLDIM) {
            empty_tensor._inplace_reshape(input_shape);
        } else {
            for (int a : axis) {
                empty_tensor = empty_tensor._expand_dims_inplace(a);
            }
        }
    }
    return empty_tensor;
}

Tensor min(const Tensor& tensor1, int axis, bool keepdims) {
    std::vector<long> axes = {axis};
    return min(tensor1, axes, keepdims);
}
Tensor min(const Tensor& tensor1, std::vector<long> axis, bool keepdims) {
    Tensor empty_tensor;
    TensorShape input_shape = tensor1.get_shape();
    axis = process_axes(tensor1.get_ndim(), axis);

    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Min(axis, keepdims))
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    // Tensor empty_tensor = empty_scalar(tensor1.get_dtype());

    auto new_shape = shape_process(tensor1, axis);
    empty_tensor = zeros(TensorShape(new_shape), tensor1.get_dtype());

    sail::internal::min_stub(tensor1, axis, empty_tensor);
    if (keepdims) {
        if (axis[0] == NULLDIM) {
            empty_tensor._inplace_reshape(input_shape);
        } else {
            for (int a : axis) {
                empty_tensor = empty_tensor._expand_dims_inplace(a);
            }
        }
    }
    return empty_tensor;
}

Tensor max(const Tensor& tensor1, int axis, bool keepdims) {
    std::vector<long> axes = {axis};
    return max(tensor1, axes, keepdims);
}
Tensor max(const Tensor& tensor1, std::vector<long> axis, bool keepdims) {
    Tensor empty_tensor;
    TensorShape input_shape = tensor1.get_shape();
    axis = process_axes(tensor1.get_ndim(), axis);

    if (tensor1.requires_grad) {
        TensorVector vec;
        vec.emplace_back(tensor1);
        empty_tensor =
            (new autograd::Max(axis, keepdims))
                ->apply(vec);  //{std::make_shared<Tensor>(tensor1)});
        return empty_tensor;
    }
    // Tensor empty_tensor = empty_scalar(tensor1.get_dtype());

    auto new_shape = shape_process(tensor1, axis);
    empty_tensor = zeros(TensorShape(new_shape), tensor1.get_dtype());

    sail::internal::max_stub(tensor1, axis, empty_tensor);
    if (keepdims) {
        if (axis[0] == NULLDIM) {
            empty_tensor._inplace_reshape(input_shape);
        } else {
            for (int a : axis) {
                empty_tensor = empty_tensor._expand_dims_inplace(a);
            }
        }
    }
    return empty_tensor;
}

}  // namespace ops
}  // namespace sail
