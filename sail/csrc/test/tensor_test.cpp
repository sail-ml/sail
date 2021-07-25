#include "core/Tensor.h"
#include "core/exception.h"
#include "core/numeric.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "gtest/gtest.h"

#include <iostream>

using namespace sail;
TEST(TensorTest, BodyCount) {
    sail::Tensor x =
        sail::random::uniform(sail::TensorShape({4, 5, 3, 2}), 0, 1);
    sail::Tensor y = x;
    sail::Tensor z = y;

    ASSERT_EQ(x.get_body_ref_count(), 3);
    ASSERT_EQ(y.get_body_ref_count(), 3);
    ASSERT_EQ(z.get_body_ref_count(), 3);
}

TEST(TensorTest, SetShape) {
    sail::Tensor x = sail::random::uniform(sail::TensorShape({1, 2, 3}), 0, 1);
    sail::TensorShape sh = sail::TensorShape({3, 2, 1});
    x.set_shape(sh);

    ASSERT_EQ(x.get_shape().shape, sh.shape);
}
TEST(TensorTest, Numeric) {
    sail::Tensor x = sail::random::uniform(sail::TensorShape({1, 2, 3}), 0, 1);
    sail::Tensor y = x + 1;
}

TEST(TensorTest, sum) {
    sail::Tensor x = sail::random::uniform(sail::TensorShape({2, 2, 3}), 0, 1);
    auto y = x.sum();
    auto z = sail::ops::sum(x);

    auto y_d = y.get<double>();
    auto z_d = z.get<double>();

    ASSERT_EQ(y_d, z_d);
}

TEST(TensorTest, iadd) {
    sail::Tensor x = sail::random::uniform(sail::TensorShape({2, 2, 3}), 0, 1);
    auto em = sail::ops::max(x);
    x += 1.0;
    auto pm = sail::ops::max(x);
}

TEST(TensorTest, iadd2) {
    sail::Tensor x = sail::random::uniform(sail::TensorShape({2, 2, 3}), 0, 1);
    auto em = sail::ops::max(x);
    x += x;
    auto pm = sail::ops::max(x);

    ASSERT_EQ(pm.get<double>(), em.get<double>() * 2);
}

TEST(TensorTest, ndim) {
    auto x = sail::random::uniform(sail::TensorShape({2, 2, 3}), 0, 1);
    ASSERT_EQ(x.ndim(), 3);

    x = sail::random::uniform(sail::TensorShape({20}), 0, 1);
    ASSERT_EQ(x.ndim(), 1);

    x = sail::random::uniform(sail::TensorShape({2, 1, 3}), 0, 1);
    ASSERT_EQ(x.ndim(), 3);

    x = sail::random::uniform(sail::TensorShape({2, 2}), 0, 1);
    ASSERT_EQ(x.ndim(), 2);
}

TEST(TensorTest, set_view) {
    auto x = sail::random::uniform(sail::TensorShape({2, 2, 3}), 0, 1);
    x.set_view();

    ASSERT_TRUE(x.is_view());
}

TEST(TensorTest, transpose1) {
    auto x = sail::random::uniform(sail::TensorShape({1, 2, 3}), 0, 1);
    x = x.transpose();
    std::vector<long> shape = {3, 2, 1};

    ASSERT_EQ(x.get_shape().shape, shape);
}

TEST(TensorTest, transpose2) {
    auto x = sail::random::uniform(sail::TensorShape({1, 2, 3}), 0, 1);
    x = x.transpose({1, 0, 2});
    std::vector<long> shape = {2, 1, 3};

    ASSERT_EQ(x.get_shape().shape, shape);
}

TEST(TensorTest, inplace_reshape_fail) {
    auto x = sail::random::uniform(sail::TensorShape({1, 2, 3}), 0, 1);
    ASSERT_THROW(x._inplace_reshape(sail::TensorShape({2, 3, 5})),
                 DimensionError);

    x = sail::random::uniform(sail::TensorShape({30}), 0, 1);
    ASSERT_THROW(x._inplace_reshape(sail::TensorShape({2})), DimensionError);

    x = sail::random::uniform(sail::TensorShape({13}), 0, 1);
    ASSERT_THROW(x._inplace_reshape(sail::TensorShape({20})), DimensionError);

    x = sail::random::uniform(sail::TensorShape({1, 2, 3, 2, 1}), 0, 1);
    ASSERT_THROW(x._inplace_reshape(sail::TensorShape({1, 2, 4, 2, 1})),
                 DimensionError);

    x = sail::random::uniform(sail::TensorShape({100, 2, 3}), 0, 1);
    ASSERT_THROW(x._inplace_reshape(sail::TensorShape({2, 3, 5})),
                 DimensionError);

    x = sail::random::uniform(sail::TensorShape({1, 22, 30}), 0, 1);
    ASSERT_THROW(x._inplace_reshape(sail::TensorShape({25})), DimensionError);
}

TEST(TensorTest, index_fail) {
    auto x = sail::random::uniform(sail::TensorShape({1, 2, 3}), 0, 1);

    ASSERT_THROW(x[0][0][0][0][0], SailCError);
}
TEST(TensorTest, get_fail) {
    auto x = sail::random::uniform(sail::TensorShape({1, 2, 3}), 0, 1);

    ASSERT_THROW(x.get<long>(), SailCError);
}

TEST(TensorTest, slice_empty) {
    auto x = sail::random::uniform(sail::TensorShape({2, 2, 3}), 0, 1);
    std::vector<long> empty = {};
    auto y = x.slice(sail::Slice(empty));

    auto xx = x.sum();
    auto yy = y.sum();

    ASSERT_EQ(xx.get<double>(), yy.get<double>());
}

TEST(TensorTest, constructor) {
    auto x = sail::random::uniform(sail::TensorShape({2, 2, 3}), 0, 1);
    auto y = sail::Tensor(x, x.requires_grad);
}

TEST(TensorShapeTest, next) {
    auto x = sail::random::uniform(sail::TensorShape({1, 20, 3}), 0, 1);
    auto y = ops::broadcast_to(x, TensorShape({20, 20, 3}));

    auto z = y.get_shape();
    for (int i = 0; i < 20 * 20 * 3; i++) {
        z.next(1);
    }
}

TEST(TensorShapeTest, next2) {
    auto x = sail::random::uniform(sail::TensorShape({1, 20, 1, 1}), 0, 1);
    auto y = ops::broadcast_to(x, TensorShape({20, 20, 3, 4}));

    auto z = y.get_shape();
    for (int i = 0; i < 20 * 20 * 3 * 2; i++) {
        z.next(2);
    }
}

TEST(TensorShapeTest, next_single_value) {
    auto x = sail::random::uniform(sail::TensorShape({1}), 0, 1);

    auto z = x.get_shape();
    for (int i = 0; i < 1; i++) {
        z.next(1);
    }
}
TEST(TensorShapeTest, next_flat) {
    auto x = sail::random::uniform(sail::TensorShape({10}), 0, 1);

    auto z = x.get_shape();
    for (int i = 1; i < 10; i++) {
        ASSERT_EQ(z.next(1), i);
    }
}

TEST(TensorShapeTest, compare) {
    auto x = sail::random::uniform(sail::TensorShape({10}), 0, 1);
    auto y = sail::random::uniform(sail::TensorShape({10}), 0, 1);
    auto z = sail::random::uniform(sail::TensorShape({1}), 0, 1);

    ASSERT_TRUE(x.get_shape() == y.get_shape());
    ASSERT_FALSE(z.get_shape() == y.get_shape());
}

TEST(TensorShapeTest, getString) {
    std::vector<long> emp = {};
    auto x = sail::TensorShape(emp);

    ASSERT_EQ(x.get_string(), "()");
}
