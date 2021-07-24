#include <algorithm>
#include <cstdint>
#include <iostream>
#include <tuple>

#include <bitset>
#include <sstream>
#include "LinalgOps.h"
#include "Tensor.h"
#include "autograd/autograd.h"
#include "dtypes.h"
#include "exception.h"
#include "kernels/Kernel.h"
#include "utils.h"

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

std::tuple<LongVec, TensorShape> GetTensorDotRollAxes(
    const TensorShape& shape, const LongVec& reduce_axes,
    bool reduced_axes_first) {
    bool to_reduce[25]{};
    LongVec remain_dims;
    LongVec remain_dims2;
    LongVec roll_axes;
    LongVec roll_axes2;
    LongVec not_in;
    for (auto i : sail::irange(0, (int)reduce_axes.size())) {
        to_reduce[reduce_axes[i]] = true;
    }

    for (auto i : sail::irange((long)0, shape.ndim())) {
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

Tensor tensordot(const Tensor& input1, const Tensor& input2, LongVec t1_dim,
                 LongVec t2_dim) {
    if (t1_dim.size() != t2_dim.size()) {
        throw SailCError("Size of axes must match");
    }

    Tensor t1 = input1;
    Tensor t2 = input2;

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

    TensorShape dot_shape =
        TensorShape({a_remain_total_size, b_remain_total_size});
    Tensor dot_out = zeros(dot_shape, dt);
    t1 = t1.cast(dt);
    t2 = t2.cast(dt);

    sail::internal::matmul_stub(t1.transpose(a_roll_axes).reshape(a_shape),
                                t2.transpose(b_roll_axes).reshape(b_shape),
                                dot_out, true, "N", "N");
    LongVec out_shape = a_remain_dims.shape;
    std::copy(b_remain_dims.shape.begin(), b_remain_dims.shape.end(),
              std::back_inserter(out_shape));
    TensorShape ret_shape = TensorShape(out_shape);
    dot_out._inplace_reshape(ret_shape);
    return dot_out;
}

Tensor matmul(const Tensor& t1, const Tensor& t2, std::string trans_a,
              std::string trans_b) {
    bool prepend = false;
    bool postpend = false;

    auto t1_e = t1;
    auto t2_e = t2;

    if (t1_e.requires_grad || t2_e.requires_grad) {
        TensorVector vec;
        vec.emplace_back(t1_e);
        vec.emplace_back(t2_e);
        Tensor empty_tensor =
            (new autograd::Matmul(trans_a, trans_b))->apply(vec);
        return empty_tensor;
    }

    if (t1_e.is_single() && t2_e.is_single()) {
        THROW_ERROR_DETAILED(SailCError, "Cannot pass single values to matmul");
    }

    if (t1_e.get_ndim() == 1) {
        t1_e = t1_e.expand_dims(0);  // NOLINT
        prepend = true;
    } else if (t2_e.get_ndim() == 1) {
        t2_e = t2_e.expand_dims(1);  // NOLINT
        postpend = true;
    }

    if (t1_e.get_ndim() != t2_e.get_ndim()) {
        if (!(t1_e.get_ndim() == 1 && t2_e.get_ndim() == 2) &&
            !(t2_e.get_ndim() == 1 && t1_e.get_ndim() == 2)) {
            THROW_ERROR_DETAILED(
                SailCError, "Incorrect number of dimensions passed to matmul");
        }
    }

    long inner_1, inner_2;

    if (t1_e.get_ndim() == 1) {
        inner_1 = t1_e.get_shape()[0];
    } else if (trans_a == NO_TRANS) {
        inner_1 = t1_e.get_shape()[1];
    } else {
        inner_1 = t1_e.get_shape()[0];
    }

    if (t2_e.get_ndim() == 1) {
        inner_2 = t2_e.get_shape()[0];
    } else if (trans_b == NO_TRANS) {
        inner_2 = t2_e.get_shape()[0];
    } else {
        inner_2 = t2_e.get_shape()[1];
    }

    SAIL_CHECK(inner_1 == inner_2, "Inner dimensions must match");

    Tensor t1_, t2_;

    if (t1_e.is_view()) {
        t1_ = clone(t1_e);
    } else {
        t1_ = t1_e;
    }
    if (t2_e.is_view()) {
        t2_ = clone(t2_e);
    } else {
        t2_ = t2_e;
    }
    Dtype dt = promote_dtype(t1_.get_dtype(), t2_.get_dtype());

    TensorSize new_shape;
    TensorSize s1 = t1_.get_shape().shape;
    TensorSize s2 = t2_.get_shape().shape;

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

    t1_ = t1_.cast(dt);
    t2_ = t2_.cast(dt);

    sail::internal::matmul_stub(t1_, t2_, empty_tensor, true, trans_a, trans_b);

    if (prepend || postpend) {
        empty_tensor._inplace_reshape(TensorShape({empty_tensor.numel()}));
    }

    return empty_tensor;
}

Tensor addmm(const Tensor& t1, const Tensor& t2, const Tensor& add) {
    if (t1.requires_grad || t2.requires_grad || add.requires_grad) {
        TensorVector vec;
        vec.emplace_back(t1);
        vec.emplace_back(t2);
        vec.emplace_back(add);
        Tensor empty_tensor = (new autograd::AddMM())->apply(vec);
        return empty_tensor;
    }

    if (t1.is_single() && t2.is_single()) {
        THROW_ERROR_DETAILED(SailCError, "Cannot pass single values to matmul");
    }

    if (t1.get_ndim() != t2.get_ndim()) {
        throw SailCError("Number of dimensions must match");
    }

    if (t1.get_shape().shape[1] != t2.get_shape().shape[0]) {
        throw SailCError("Inner dimensions must match");
    }

    Dtype dt = promote_dtype(t1.get_dtype(), t2.get_dtype());
    Tensor t1_in = t1;
    Tensor t2_in = t2;

    if (t1.is_view() || t2.is_view()) {
        t1_in = clone(t1);
        t2_in = clone(t2);
    }

    auto t1_casted = t1.cast(dt);
    auto t2_casted = t2.cast(dt);
    auto add_casted = add.cast(dt);

    TensorSize new_shape = {t1_casted.get_shape().shape[0],
                            t2_casted.get_shape().shape[1]};
    TensorShape s = TensorShape(new_shape);
    Tensor add_ = ops::broadcast_to(add, s);
    Tensor empty_tensor = clone(add_);

    sail::internal::matmul_stub(t1_casted, t2_casted, empty_tensor, false, "N",
                                "N");

    return empty_tensor;
}
}  // namespace ops

}  // namespace sail
