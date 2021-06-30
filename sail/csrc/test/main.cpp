#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/factories.h"
#include "core/tensor_shape.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}