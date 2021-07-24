#include "gtest/gtest.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/Tensor.h"
#include "core/exception.h"
#include "core/dtypes.h"
#include "core/tensor_shape.h"

#include <iostream>

TEST(CompareTest, LT) {
    
    float arr[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    bool correct[10] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    sail::Tensor t = sail::from_data((void*)arr, Dtype::sFloat32, sail::TensorShape({10}));
    auto y = 3 > t;

    for (int i = 0; i < 10; i++) {
        auto v = ((bool*)(y.get_data()))[i];
        ASSERT_EQ(correct[i], v);
    }

}
TEST(CompareTest, GT) {
    
    float arr[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    bool correct[10] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    sail::Tensor t = sail::from_data((void*)arr, Dtype::sFloat32, sail::TensorShape({10}));
    auto y = 3 < t;

    for (int i = 0; i < 10; i++) {
        auto v = ((bool*)(y.get_data()))[i];
        ASSERT_EQ(correct[i], v);
    }

}

TEST(CompareTest, LTE) {
    
    float arr[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    bool correct[10] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
    sail::Tensor t = sail::from_data((void*)arr, Dtype::sFloat32, sail::TensorShape({10}));
    auto y = 3 >= t;

    for (int i = 0; i < 10; i++) {
        auto v = ((bool*)(y.get_data()))[i];
        ASSERT_EQ(correct[i], v);
    }

}
TEST(CompareTest, GTE) {
    
    float arr[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    bool correct[10] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1};
    sail::Tensor t = sail::from_data((void*)arr, Dtype::sFloat32, sail::TensorShape({10}));
    auto y = 3 <= t;

    for (int i = 0; i < 10; i++) {
        auto v = ((bool*)(y.get_data()))[i];
        ASSERT_EQ(correct[i], v);
    }

}

TEST(CompareTest, NEQ) {
    
    float arr[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    bool correct[10] = {1, 1, 1, 0, 1, 1, 1, 1, 1, 1};
    sail::Tensor t = sail::from_data((void*)arr, Dtype::sFloat32, sail::TensorShape({10}));
    auto y = 3 != t;

    for (int i = 0; i < 10; i++) {
        auto v = ((bool*)(y.get_data()))[i];
        ASSERT_EQ(correct[i], v);
    }

}
TEST(CompareTest, EQ) {
    
    float arr[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    bool correct[10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    sail::Tensor t = sail::from_data((void*)arr, Dtype::sFloat32, sail::TensorShape({10}));
    auto y = 3 == t;

    for (int i = 0; i < 10; i++) {
        auto v = ((bool*)(y.get_data()))[i];
        ASSERT_EQ(correct[i], v);
    }

}
