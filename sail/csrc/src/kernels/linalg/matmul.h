#pragma once
#define OMP_MIN_VALUE 512  // should probably find a real number ot use

// #include <cblas.h>
// #include <immintrin.h>
#include <iostream>

#include "../../Tensor.h"
#include "../base.h"
#include "../unary.h"

#ifdef MKL
#include <mkl.h>
#include <omp.h>
#else
extern "C" {
#include <cblas.h>
}
#include <omp.h>
#endif

#define TRANS "T"
#define NO_TRANS "N"
#define CONJ_TRANS "C"

#include <chrono>
using namespace std::chrono;

namespace sail {

class DotTTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
                 bool empty = false, std::string trans_a = "N",
                 std::string trans_b = "N") {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            auto name = decltype(pt)::GetName();

            std::vector<long> t1_shape = t1.get_shape().shape;
            std::vector<long> t2_shape = t2.get_shape().shape;

            int beta;
            if (empty) {
                beta = 0.0;
            } else {
                beta = 1.0;
            }
            int M_b = t1_shape[0];  // ROWS IN A
            int K_b = t1_shape[1];  // COLS IN A AND ROWS IN B
            int N_b = t2_shape[1];  // COLS IN B

            int M = M_b;
            int N = N_b;
            int K = K_b;

            int lda = t1_shape[1];
            int ldb = t2_shape[1];
            int ldc = t2_shape[1];

            CBLAS_TRANSPOSE cblas_transa;
            CBLAS_TRANSPOSE cblas_transb;

            if (trans_a == TRANS) {
                cblas_transa = CblasTrans;
                M = t1_shape[1];
                K = t1_shape[0];
            } else if (trans_a == NO_TRANS) {
                cblas_transa = CblasNoTrans;
            } else {
                cblas_transa = CblasConjTrans;
            }

            if (trans_b == TRANS) {
                cblas_transb = CblasTrans;
                N = t2_shape[0];
                K = t2_shape[1];
            } else if (trans_b == NO_TRANS) {
                cblas_transb = CblasNoTrans;
            } else {
                cblas_transb = CblasConjTrans;
            }

            if (name == "float64") {
                using T = double;
                cblas_dgemm(CblasRowMajor, cblas_transa, cblas_transb, M, N, K,
                            1, (T*)t1.get_data(), lda, (T*)t2.get_data(), ldb,
                            beta, (T*)out_tensor.get_data(), N);

            } else if (name == "float32") {
                using T = float;
                cblas_sgemm(CblasRowMajor, cblas_transa, cblas_transb, M, N, K,
                            1, (T*)t1.get_data(), lda, (T*)t2.get_data(), ldb,
                            beta, (T*)out_tensor.get_data(), N);
                //
            } else {
                using T = typename decltype(pt)::type;

                // if (trans_a == TRANS) {
                //     t1 = clone(t1.transpose({1, 0}));
                // }
                // if (trans_b == TRANS) {
                //     t2 = clone(t2.transpose({1, 0}));
                // }

                T* matA = (T*)t1.get_data();
                T* matB = (T*)t2.get_data();
                T* matC = (T*)out_tensor.get_data();

                // TODO: TRANS

                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        T sum = 0.0;
                        for (int k = 0; k < K; k++)
                            sum = sum + matA[i * K + k] * matB[k * N + j];
                        matC[i * N + j] = sum;
                    }
                }
            }
        });
    }
};  // namespace sail

}  // namespace sail