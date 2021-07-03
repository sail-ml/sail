#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "core/slice.h"

int main(int argc, char **argv) {
    // ::testing::InitGoogleTest(&argc, argv);
    // return RUN_ALL_TESTS();
    sail::Tensor t1 = sail::random::uniform(sail::TensorShape({1, 4, 4}), 3, 3);
    // float arr[16];
    // for (int i = 0; i < 16; i++) {
    //     arr[i] = (float)i;
    // }
    // sail::Tensor t1 = sail::from_data((void*)arr, Dtype::sFloat32, sail::TensorShape({1, 4, 4}));
    sail::Tensor t2 = sail::random::uniform(sail::TensorShape({1, 1, 3, 3}), 4,4);
    sail::Tensor x = sail::ops::conv2d(t1,t2, {2, 2});
    std::cout << x << std::endl;
    // std::cout << t1 << std::endl;

    // sail::Slice s1 = sail::Slice({{1, 3}, {1, 3}, {1, 3}});

    // // std::cout << t1.slice(s1) << std::endl;
    // // sail::Tensor slice = t1.slice(s1);
    // // t1.slice(s1).assign(t2);
    // // std::cout << t1 << std::endl;
    // // std::cout << " " << std::endl;
    // // std::cout << " " << std::endl;
    // sail::Tensor x = sail::ops::pad(t1, {{1, 1}, {1, 1}, {1, 1}});
    // std::cout << x << std::endl;
    return 0;
}