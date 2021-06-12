#include "gtest/gtest.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/Tensor.h"
#include "core/error.h"
#include "core/dtypes.h"
#include "core/tensor_shape.h"

#include <iostream>

TEST(FactoriesTest, OneHot) {
    
    int32_t arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    sail::Tensor integer_tensor = sail::from_data((void*)arr, Dtype::sInt32, sail::TensorShape({10}));

    sail::Tensor one_hot_tensor = sail::one_hot(integer_tensor, 10);

    for (int i = 0; i < 10; i++) {
        sail::Tensor z = one_hot_tensor[i];
        sail::Tensor check = integer_tensor[i];
        
        int32_t check_data = ((int32_t*)check.get_data())[0];
        sail::Tensor z2 = z[check_data];
        int32_t z_data = ((int32_t*)z2.get_data())[0];
        ASSERT_EQ(z_data, 1);
    }

}
TEST(FactoriesTest, OneHotThrow) {
    
    float arr[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    sail::Tensor t = sail::from_data((void*)arr, Dtype::sFloat32, sail::TensorShape({10}));

    ASSERT_THROW(sail::one_hot(t, 10), SailCError);

}

TEST(FactoriesTest, Ones) {

    sail::Tensor one_tensor = sail::ones(sail::TensorShape({32, 32, 32}), Dtype::sInt32);
    int32_t* data = (int32_t*)one_tensor.get_data();
    for (int i = 0; i < 32 * 32 * 32; i++) {
        ASSERT_EQ(data[i], 1);
    }

}