#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <unistd.h>
#include <utility>
#include "gtest/gtest.h"
#include "core/Tensor.h"
#include "core/modules/modules.h"
#include "core/factories.h"
#include "core/ops/ops.h"
#include "core/tensor_shape.h"
#include "core/slice.h"
#include "core/kernels/Kernel.h"
#include "core/onednn/pooling.h"

using namespace sail;
int main(int argc, char **argv) {
    // ::testing::InitGoogleTest(&argc, argv);
    // return RUN_ALL_TESTS();

    auto a = sail::random::uniform(TensorShape({1, 1, 32, 32}), 0, 1);

    auto l1 = sail::modules::Conv2D(1, 16, 3, 1, "same");
    // auto l2 = sail::modules::Conv2D(16, 1, 3, 1, "same");

    auto x = l1.forward(a);
    // auto y = l2.forward(x);

    auto s = sail::ops::sum(x);
    s.backward();

    std::cout << l1.weights.get_grad() << std::endl;

    // for (int i = 0; i < 100; i++) {
        // x.forward(a);
    // }

     return 0;
}