#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include "core/Tensor.h"
#include "core/factories.h"
#include "core/kernels/Kernel.h"
#include "core/modules/modules.h"
#include "core/onednn/pooling.h"
#include "core/ops/ops.h"
#include "core/slice.h"
#include "core/tensor_shape.h"
#include "gtest/gtest.h"

using namespace sail;
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

    return 0;
}