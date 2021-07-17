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

namespace sail {

namespace ops {
using TensorVector = std::vector<Tensor>;

// std::tuple<LongVec, TensorShape> GetTensorDotRollAxes(
//     const TensorShape& shape, const LongVec& reduce_axes,
//     bool reduced_axes_first) {
//     bool to_reduce[25]{};  // Initialized with false.
//     LongVec remain_dims;
//     LongVec remain_dims2;
//     LongVec roll_axes;
//     LongVec roll_axes2;
//     LongVec not_in;
//     for (int8_t i = 0; i < reduce_axes.size(); ++i) {
//         to_reduce[reduce_axes[i]] = true;
//     }

//     // There are two steps:
//     // A. Insert axes to be reduced to roll_axes.
//     // B. Insert non-reduced axes to roll_axes.
//     // The order of these steps depends on reduced_axes_first.
//     for (int step = 0; step < 2; ++step) {
//         if ((step == 0) == reduced_axes_first) {
//             // Step A.
//             for (int8_t i = 0; i < shape.ndim(); ++i) {
//                 if (to_reduce[i]) {
//                     roll_axes.emplace_back(i);
//                 }
//             }
//         } else {
//             // Step B.
//             for (int8_t i = 0; i < shape.ndim(); ++i) {
//                 if (!to_reduce[i]) {
//                     roll_axes.emplace_back(i);
//                     remain_dims.emplace_back(shape[i]);
//                 }
//             }
//         }
//     }

//     return std::make_tuple(roll_axes, TensorShape(remain_dims));
// }

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

static inline std::bitset<64> dim_list_to_bitset(LongVec dims, int64_t ndims) {
    std::bitset<64> seen;
    for (size_t i = 0; i < dims.size(); i++) {
        size_t dim = dims[i];
        if (dims[i] < 0) {
            dim = dims[i] + ndims;
        }
        seen[dim] = true;
    }
    return seen;
}

Tensor tensordot(const Tensor& input1, const Tensor& input2, LongVec t1_dim,
                 LongVec t2_dim) {
    if (t1_dim.size() != t2_dim.size()) {
        throw SailCError("Size of axes must match");
    }

    // int64_t csize = 1;  // total size of the contracted dimensions
    Tensor t1 = input1;
    Tensor t2 = input2;
    // for (int i = 0; i < t1_dim.size(); i++) {
    //     int s1 = input1.get_shape()[t1_dim[i]];
    //     int s2 = input2.get_shape()[t2_dim[i]];
    //     if (s2 == 1) {  // broadcasted dimensions can be summed right away
    //         t1 = ops::sum(t1, t1_dim[i], true);
    //         //   t1 = t1.sum(dims1[i], true);
    //     } else if (s1 == 1) {
    //         t2 = ops::sum(t2, t2_dim[i], true);

    //         //   t2 = t2.sum(dims2[i], true);
    //     } else {
    //         //   TORCH_CHECK(s1 == s2, "contracted dimensions need to match,
    //         but
    //         //   first has size ", s1, " in dim ", dims1[i],
    //         //            " and second has size ", s2, " in dim ", dims2[i]);
    //         csize *= s1;
    //     }
    // }

    // auto cdims1 = dim_list_to_bitset(t1_dim, input1.ndim());
    // auto cdims2 = dim_list_to_bitset(t2_dim, input2.ndim());
    // std::vector<int64_t> p1, p2,
    //     rsizes;  // p1, p2: input permutations, rsizes: sizes of the result
    // p1.reserve(input1.ndim());
    // p2.reserve(input2.ndim());
    // rsizes.reserve(input1.ndim() + input2.ndim() - (int64_t)t1_dim.size());
    // int64_t size1 = 1;  // number of non-contracted elements in input1
    // int64_t size2 = 1;  // number of non-contracted elements in input2

    // // fill the permutations and compute sizes
    // for (int i = 0; i < input1.ndim(); i++) {
    //     //   for (const auto i : c10::irange(input1.dim())) {
    //     if (!cdims1[i]) {
    //         p1.emplace_back(i);
    //         size1 *= t1.get_shape()[i];
    //         rsizes.emplace_back(t1.get_shape()[i]);
    //     }
    // }
    // for (const auto x : t1_dim) {
    //     p1.emplace_back(x);
    // }
    // for (const auto x : t2_dim) {
    //     p2.emplace_back(x);
    // }
    // for (int i = 0; i < input2.ndim(); i++) {
    //     //   for (const auto i : c10::irange(input2.dim())) {
    //     if (!cdims2[i]) {
    //         p2.emplace_back(i);
    //         size2 *= t2.get_shape()[i];
    //         rsizes.emplace_back(t2.get_shape()[i]);
    //     }
    // }
    // // permut and reshape for matrix multiplication
    // // t1 = t1.permute(p1).reshape({size1, csize});
    // // t2 = t2.permute(p2).reshape({csize, size2});

    // t1 = t1.transpose(p1).reshape(TensorShape({size1, csize}));
    // t2 = t2.transpose(p2).reshape(TensorShape({csize, size2}));

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
    dot_out._inplace_reshape(ret_shape);
    // ret_shape.reset();
    return dot_out;

    // auto out_ = matmul(t1, t2, "N", "N");
    // out_._inplace_reshape(TensorShape({rsizes}));
    // return matmul(t1, t2, "N", "N");
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
    Tensor t1_, t2_;

    if (t1.is_view()) {
        t1_ = clone(t1);
    } else {
        t1_ = t1;
    }
    if (t2.is_view()) {
        t2_ = clone(t2);
    } else {
        t2_ = t2;
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
