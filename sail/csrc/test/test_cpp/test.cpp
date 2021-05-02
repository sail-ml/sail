#include "../../src/Tensor.h"
#include "../../src/tensor_shape.h"
#include "../../src/dtypes.h"
#include "../../src/ops/ops.h"

#include <iostream>
#include <ostream>
#include <cblas.h>

#include <gtest/gtest.h>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

#define MAX_VAL 320000

// Demonstrate some basic assertions.
TEST(SailTest, FreeTest) {
  // Expect two strings not to be equal.
    double x[MAX_VAL];
    double y[MAX_VAL];
    int ndim = 1;
    Dtype dt = Dtype::sFloat64;
    TensorSize st = {8};
    TensorSize sh = {MAX_VAL};

    sail::TensorShape sp = sail::TensorShape(sh, st);

    for (int i = 0; i < MAX_VAL; i++) {
        x[i] = 2.21;
        y[i] = 3.21;
    }

    void* xt = static_cast<void*>(x);
    void* yt = static_cast<void*>(y);

    sail::Tensor t1 = sail::Tensor(ndim, xt, dt, sp);
    sail::Tensor t2 = sail::Tensor(ndim, yt, dt, sp);

    sail::Tensor t3 = t1 + t2;

    t1.free();
    t2.free();
    t3.free();

    ASSERT_EQ(t1.data, NULL);
    ASSERT_EQ(t2.data, NULL);
}

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

TEST(SailTest, CBlas) {
    int i=0;
    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};  // 2x3       
    double B[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};  // 3x2
    double C[4] = {0, 0, 0, 0}; // 2x2

    // M = cols of A
    // N = rows of B
    // K = rows of A and cols of B

    int M = 2;
    int N = 2;
    int K = 3;

    // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 3, B, 3,1,C,3);
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans, // TRANSA
        CblasNoTrans, // TRANSB
        M, // M (rows of matrix A and C)
        N, // M (cols of matrix B and C)
        K, // K (cols of matrix A and rows of B)
        1, // Alpha (multiply A by)
        A, // matrix A
        K, // LDA (first dimension of A to be called on? I think it is the stride of the row)
        B, // matrix B
        N, // LDB (first dimension of B to be called on? I think it is the stride of the row)
        1, // Beta (multiply B by)
        C, // matrix C (res)
        N // LDC (same as LDB and LDA)
    );
    for(i=0; i<4; i++)
        printf("%lf ", C[i]);
    printf("\n");
}