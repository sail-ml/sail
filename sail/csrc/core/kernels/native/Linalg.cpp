// allow-no-header

#include "kernels/Linalg.h"
#include <iostream>
#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"

extern "C" {
#include <cblas.h>
}
#include <omp.h>

namespace sail {

namespace internal {

namespace {

void matmul_kernel(const Tensor& t1, const Tensor& t2, Tensor& out_tensor,
                   bool empty = false, std::string trans_a = "N",
                   std::string trans_b = "N") {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        auto name = decltype(pt)::GetName();

        SAIL_CHECK(!t1.is_view(), "Cannot pass views to matmul");
        SAIL_CHECK(!t2.is_view(), "Cannot pass views to matmul");

        std::vector<long> t1_shape = t1.get_shape().shape;
        std::vector<long> t2_shape = t2.get_shape().shape;

        int beta;
        if (empty) {
            beta = 0.0;
        } else {
            beta = 1.0;
        }
        int M_b = t1_shape[0];
        int K_b = t1_shape[1];
        int N_b = t2_shape[1];

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
            cblas_dgemm(CblasRowMajor, cblas_transa, cblas_transb, M, N, K, 1,
                        (T*)t1.get_data(), lda, (T*)t2.get_data(), ldb, beta,
                        (T*)out_tensor.get_data(), N);

        } else if (name == "float32") {
            using T = float;
            cblas_sgemm(CblasRowMajor, cblas_transa, cblas_transb, M, N, K, 1,
                        (T*)t1.get_data(), lda, (T*)t2.get_data(), ldb, beta,
                        (T*)out_tensor.get_data(), N);
        } else {
            using T = typename decltype(pt)::type;

            T* matA = (T*)t1.get_data();
            T* matB = (T*)t2.get_data();
            T* matC = (T*)out_tensor.get_data();

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

}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(matmul_stub, &matmul_kernel);

}  // namespace internal

}  // namespace sail