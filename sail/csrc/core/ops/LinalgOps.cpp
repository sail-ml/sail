#include <algorithm>
#include <cstdint>
#include <iostream>
#include <tuple>

#include "LinalgOps.h"
#include "Tensor.h"
#include "autograd/autograd.h"
#include "dtypes.h"
#include "exception.h"
#include "kernels/Kernel.h"

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

std::tuple<LongVec, TensorShape> GetTensorDotRollAxes(
    const TensorShape& shape, const LongVec& reduce_axes,
    bool reduced_axes_first) {
    bool to_reduce[25]{};  // Initialized with false.
    LongVec remain_dims;
    LongVec remain_dims2;
    LongVec roll_axes;
    LongVec roll_axes2;
    LongVec not_in;
    for (int8_t i = 0; i < reduce_axes.size(); ++i) {
        to_reduce[reduce_axes[i]] = true;
        // gsl::at(to_reduce, reduce_axes[i]) = true;
    }

    for (int i = 0; i < shape.ndim(); i++) {
        if (std::find(reduce_axes.begin(), reduce_axes.end(), i) ==
            reduce_axes.end()) {
            not_in.push_back(i);
            remain_dims2.emplace_back(shape.shape[i]);
        }
    }

    if (reduced_axes_first) {
        roll_axes2 = reduce_axes;
        roll_axes2.insert(roll_axes2.end(), not_in.begin(), not_in.end());
    } else {
        roll_axes2 = not_in;
        roll_axes2.insert(roll_axes2.end(), reduce_axes.begin(),
                          reduce_axes.end());
    }

    return std::make_tuple(roll_axes2, TensorShape(remain_dims2));
}

Tensor tensordot(const Tensor& t1, const Tensor& t2, int axes) {
    LongVec t1_dim, t2_dim;

    for (int i = t1.get_ndim() - 1; i > (t1.get_ndim() - axes - 1); i--) {
        t1_dim.insert(t1_dim.begin(), i);
    }

    for (int i = 0; i < axes; i++) {
        t2_dim.push_back(i);
    }

    return tensordot(t1, t2, t1_dim, t2_dim);
}
Tensor tensordot(const Tensor& t1, const Tensor& t2, LongVec t1_dim,
                 LongVec t2_dim) {
    if (t1_dim.size() != t2_dim.size()) {
        throw SailCError("Size of axes must match");
    }
    int64_t axis_total_size = 1;
    int axis_ndim = t1_dim.size();
    for (int8_t i = 0; i < axis_ndim; ++i) {
        int64_t a_dim = t1.get_shape().shape[t1_dim[i]];
        axis_total_size *= a_dim;
    }

    Dtype dt = promote_dtype(t1.get_dtype(), t2.get_dtype());

    auto a_tup = GetTensorDotRollAxes(t1.get_shape(), t1_dim, false);
    auto b_tup = GetTensorDotRollAxes(t2.get_shape(), t2_dim, true);
    LongVec a_roll_axes = std::get<0>(a_tup);
    LongVec b_roll_axes = std::get<0>(b_tup);
    TensorShape a_remain_dims = std::get<1>(a_tup);
    TensorShape b_remain_dims = std::get<1>(b_tup);
    int64_t a_remain_total_size = a_remain_dims.numel();
    int64_t b_remain_total_size = b_remain_dims.numel();
    TensorShape a_shape = TensorShape({a_remain_total_size, axis_total_size});
    TensorShape b_shape = TensorShape({axis_total_size, b_remain_total_size});

    // if (a_shape.numel() != b_shape.numel()) {
    //     throw SailCError("Shape mismatch for tensordot");
    // }

    TensorShape dot_shape =
        TensorShape({a_remain_total_size, b_remain_total_size});
    Tensor dot_out = zeros(dot_shape, dt);
    t1 = t1.cast(dt);
    t2 = t2.cast(dt);
    // DotTTKernel().execute
    sail::internal::matmul_stub(t1.transpose(a_roll_axes).reshape(a_shape),
                                t2.transpose(b_roll_axes).reshape(b_shape),
                                dot_out, true, "N", "N");
    LongVec out_shape = a_remain_dims.shape;
    std::copy(b_remain_dims.shape.begin(), b_remain_dims.shape.end(),
              std::back_inserter(out_shape));
    TensorShape ret_shape = TensorShape(out_shape);
    // ret_shape.reset();
    return dot_out._inplace_reshape(ret_shape);
}

Tensor matmul(const Tensor& t1, const Tensor& t2,
              std::string trans_a = NO_TRANS, std::string trans_b = NO_TRANS) {
    // NEED TO CHECK NDIM, TYPE, AND SHAPES SO THAT IT WORKS
    // ALSO NO SCALARS

    if (t1.requires_grad || t2.requires_grad) {
        TensorVector vec;
        vec.emplace_back(t1);
        vec.emplace_back(t2);
        Tensor empty_tensor =
            (new autograd::Matmul(trans_a, trans_b))->apply(vec);
        return empty_tensor;
    }

    if (t1.is_scalar() || t2.is_scalar()) {
        THROW_ERROR_DETAILED(SailCError, "Cannot pass scalars to matmul");
    }

    if (t1.get_ndim() != t2.get_ndim()) {
        THROW_ERROR_DETAILED(SailCError, "Number of dimensions must match");
    }

    if (trans_a == NO_TRANS && trans_b == NO_TRANS) {
        if (t1.get_shape().shape[1] != t2.get_shape().shape[0]) {
            THROW_ERROR_DETAILED(SailCError, "Inner dimensions must match");
        }
    } else if (trans_a == TRANS && trans_b == NO_TRANS) {
        if (t1.get_shape().shape[0] != t2.get_shape().shape[0]) {
            THROW_ERROR_DETAILED(SailCError, "Inner dimensions must match");
        }
    } else if (trans_a == NO_TRANS && trans_b == TRANS) {
        if (t1.get_shape().shape[1] != t2.get_shape().shape[1]) {
            THROW_ERROR_DETAILED(SailCError, "Inner dimensions must match");
        }
    } else {
        if (t1.get_shape().shape[0] != t2.get_shape().shape[1]) {
            THROW_ERROR_DETAILED(SailCError, "Inner dimensions must match");
        }
    }

    if (t1.is_view() || t2.is_view()) {
        // TODO(tgs266): This needs to not use clone. Instead, switch to
        // tensorshape iterator in kernel
        t1 = clone(t1);
        t2 = clone(t2);
    }
    Dtype dt = promote_dtype(t1.get_dtype(), t2.get_dtype());

    TensorSize new_shape;
    TensorSize s1 = t1.get_shape().shape;
    TensorSize s2 = t2.get_shape().shape;

    int r = s1[0];
    if (trans_a == TRANS) {
        r = s1[1];
    }
    int c = s2[1];
    if (trans_b == TRANS) {
        c = s2[0];
    }
    new_shape.push_back(r);
    new_shape.push_back(c);
    TensorShape s = TensorShape(new_shape);
    Tensor empty_tensor = empty(0, dt, s);

    t1 = t1.cast(dt);
    t2 = t2.cast(dt);

    sail::internal::matmul_stub(t1, t2, empty_tensor, true, trans_a, trans_b);

    return empty_tensor;
}

Tensor addmm(const Tensor& t1, const Tensor& t2, const Tensor& add) {
    // NEED TO CHECK NDIM, TYPE, AND SHAPES SO THAT IT WORKS
    // ALSO NO SCALARS

    if (t1.requires_grad || t2.requires_grad || add.requires_grad) {
        TensorVector vec;
        vec.emplace_back(t1);
        vec.emplace_back(t2);
        vec.emplace_back(add);
        Tensor empty_tensor = (new autograd::AddMM())->apply(vec);
        return empty_tensor;
    }

    if (t1.is_scalar() || t2.is_scalar()) {
        throw SailCError("Cannot pass scalars to matmul");
    }

    if (t1.get_ndim() != t2.get_ndim()) {
        throw SailCError("Number of dimensions must match");
    }

    if (t1.get_shape().shape[1] != t2.get_shape().shape[0]) {
        throw SailCError("Inner dimensions must match");
    }

    if (t1.is_view() || t2.is_view()) {
        t1 = clone(t1);
        t2 = clone(t2);
    }

    Dtype dt = promote_dtype(t1.get_dtype(), t2.get_dtype());
    t1 = t1.cast(dt);
    t2 = t2.cast(dt);
    add = add.cast(dt);

    TensorSize new_shape = {t1.get_shape().shape[0], t2.get_shape().shape[1]};
    TensorShape s = TensorShape(new_shape);
    Tensor add_ = ops::broadcast_to(add, s);
    Tensor empty_tensor = clone(add_);

    sail::internal::matmul_stub(t1, t2, empty_tensor, false, "N", "N");

    return empty_tensor;
}
}  // namespace ops

}  // namespace sail
