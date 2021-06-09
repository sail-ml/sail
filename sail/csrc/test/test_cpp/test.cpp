#include "../../src/Tensor.h"
#include "../../src/loss/cross_entropy_loss.h"
#include "../../src/optimizers/optimizers.h"
#include "../../src/tensor_shape.h"
#include "../../src/factories.h"
#include "../../src/autograd/autograd.h"
#include "../../src/modules/modules.h"
#include "../../src/dtypes.h"
#include "../../src/ops/ops.h"

#include <iostream>
#include <ostream>
// #include <cblas.h>

// #include <gtest/gtest.h>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

#define MAX_VAL 320000

int main() {
    int z = 4;
    int b = 4;
    int32_t x[z];
    int32_t y[z];
    int ndim = 1;
    Dtype dt = Dtype::sInt32;
    TensorSize st = {1};
    TensorSize sh = {z};

    sail::TensorShape sp = sail::TensorShape(sh);
    sail::TensorShape sp2 = sail::TensorShape({4, 10});

    for (int i = 0; i < z; i++) {
        x[i] = 2;
    }
    x[2] = 0;

    void* xt = static_cast<void*>(x);

    sail::Tensor t1 = sail::from_data(xt, dt, sp);
    sail::Tensor t2 = sail::random::uniform(sp2, 1, 2);

    sail::loss::SoftmaxCrossEntropyLoss loss = sail::loss::SoftmaxCrossEntropyLoss();
    // std::cout << sail::ops::sum(t1, 1, true) << std::endl;
    sail::modules::Linear lay = sail::modules::Linear(10, 32, true);
    sail::modules::Linear lay2 = sail::modules::Linear(32, 4, true);
    sail::modules::Sigmoid sig = sail::modules::Sigmoid();
    // std::cout << lay.weights << std::endl;
    // for (int i = 0; i < 3; i++) {

    sail::Tensor r = lay.forward(t2);
    sail::Tensor s = sig.forward(r);
    sail::Tensor k = lay2.forward(s);
    sail::Tensor l = loss.forward(k, t1);
    // l.backward();
    // }
    
    // std::cout << t1[0] << std::endl;
    // sail::Tensor d = sail::one_hot(t1, 15);
    // std::cout << d << std::endl;

    

    // std::cout << lay.weights << std::endl;
    // std::cout << lay.weights << std::endl;
    // std::cout << t1 << std::endl;
    // std::cout << sail::ops::softmax(t1) << std::endl;

    return 0;
}
// int main() {
//     int z = 4;
//     int b = 2;
//     double x[z * b];
//     int32_t y[z];
//     int ndim = 1;
//     Dtype dt = Dtype::sInt32;
//     TensorSize st = {1};
//     TensorSize sh = {z, b};

//     sail::TensorShape sp = sail::TensorShape(sh);
//     sail::TensorShape sp2 = sail::TensorShape({z});

//     for (int i = 0; i < z; i++) {
//         x[i] = 2.21;
//         y[i] = 3;
//     }

//     void* yt = static_cast<void*>(y);

//     sail::optimizers::SGD opt = sail::optimizers::SGD((float)0.001);

//     // sail::Tensor t1 = sail::from_data(xt, dt, sp);
//     sail::Tensor t1 = sail::random::uniform(sp);
//     sail::Tensor t2 = sail::from_data(yt, dt, sp2);
//     t1.requires_grad = true;
//     // std::cout << sail::ops::sum(t1, 1, true) << std::endl;
//     sail::modules::Sigmoid lay = sail::modules::Sigmoid();

//     sail::Tensor result = lay.forward(t1);
//     result = lay.forward(t1);
//     sail::Tensor
//     // result = lay.forward(t1);
//     // result = lay.forward(t1);
//     // sail::Tensor loss_value = loss.forward(result, t2);
//     // std::cout << loss_value << std::endl;
//     // std::cout << lay.weights << std::endl;
//     // loss_value.backward();
//     // opt.update();
//     // std::cout << lay.weights << std::endl;

//     // std::cout << lay.weights << std::endl;
//     // lay.forward(t1);
    

//     // std::cout << lay.weights << std::endl;
//     // std::cout << lay.weights << std::endl;
//     // std::cout << t1 << std::endl;
//     // std::cout << sail::ops::softmax(t1) << std::endl;

//     return 0;
// }

// Demonstrate some basic assertions.
// TEST(SailTest, FreeTest) {
//   // Expect two strings not to be equal.
//     std::vector<int> a = {1, 256};
//     for (int z : a) {

//         double x[z];
//         double y[z];
//         int ndim = 1;
//         Dtype dt = Dtype::sFloat64;
//         TensorSize st = {8};
//         TensorSize sh = {z};

//         sail::TensorShape sp = sail::TensorShape(sh, st);

//         for (int i = 0; i < z; i++) {
//             x[i] = 2.21;
//             y[i] = 3.21;
//         }

//         void* xt = static_cast<void*>(x);
//         void* yt = static_cast<void*>(y);

//         sail::Tensor t1 = sail::Tensor(ndim, xt, dt, sp, false);
//         sail::Tensor t2 = sail::Tensor(ndim, yt, dt, sp, false);

//         sail::Tensor t3 = sail::ops::add(t1, t2);

//     std::cout << sail::ops::tensor_repr(t3) << std::endl;
    

//     }

// }

// TEST(SailTest, CastFloat32ToInt32) {
//     float x[MAX_VAL];
//     int ndim = 1;
//     Dtype dt = Dtype::sFloat32;
//     TensorSize st = {8};
//     TensorSize sh = {MAX_VAL};

//     float val = 2.21;

//     for (int i = 0; i < MAX_VAL; i++) {
//         x[i] = 2.21;
//     }

//     void* xt = static_cast<void*>(x);

//     sail::Tensor t1 = sail::Tensor(ndim, xt, dt, st, sh);
//     sail::Tensor t2 = t1.cast(Dtype::sInt32);

//     int32_t* t2_data = (int32_t*)(t2.data);
//     float* t1_data = (float*)(t1.data);

//     for (int i = 0; i < MAX_VAL; i++) {
//         ASSERT_EQ(t2_data[i], 2);
//         ASSERT_EQ(t1_data[i], val);
//     }

//     t1.free();
//     t2.free();

//     ASSERT_EQ(t1.data, NULL);
//     ASSERT_EQ(t2.data, NULL);
// }


// TEST(SailTest, CastInt32ToFloat64) {
//     int32_t x[MAX_VAL];
//     int ndim = 1;
//     Dtype dt = Dtype::sInt32;
//     TensorSize st = {4};
//     TensorSize sh = {MAX_VAL};

//     int32_t val = 2;
//     double check_val = 2;

//     for (int i = 0; i < MAX_VAL; i++) {
//         x[i] = 2;
//     }

//     void* xt = static_cast<void*>(x);

//     sail::Tensor t1 = sail::Tensor(ndim, xt, dt, st, sh);
//     sail::Tensor t2 = t1.cast(Dtype::sFloat64);

//     double* t2_data = (double*)(t2.data);
//     int32_t* t1_data = (int32_t*)(t1.data);

//     for (int i = 0; i < MAX_VAL; i++) {
//         ASSERT_EQ(t2_data[i], check_val);
//         ASSERT_EQ(t1_data[i], val);
//     }

//     t1.free();
//     t2.free();

//     ASSERT_EQ(t1.data, NULL);
//     ASSERT_EQ(t2.data, NULL);
// }

// TEST(SailTest, CBlas) {
//     double x[32];
//     double y[32];
//     int ndim = 1;
//     Dtype dt = Dtype::sFloat64;
//     TensorSize st = {8};
//     TensorSize sh = {32};

//     sail::TensorShape sp = sail::TensorShape(sh, st);

//     for (int i = 0; i < 32; i++) {
//         x[i] = 2.21;
//         y[i] = 3.21;
//     }

//     void* xt = static_cast<void*>(x);
//     void* yt = static_cast<void*>(y);

//     sail::Tensor t1 = sail::Tensor(ndim, xt, dt, sp, true);
//     sail::Tensor t2 = sail::Tensor(ndim, yt, dt, sp, true);

//     std::cout << sail::ops::tensor_repr(t1) << std::endl;

//     t1.free();
//     t2.free();
// }