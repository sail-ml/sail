#pragma once
#define OMP_MIN_VALUE 512  // should probably find a real number ot use

#include <cblas.h>
#include <immintrin.h>
#include <iostream>

#include "../../Tensor.h"
#include "../base.h"
#include "../unary.h"

namespace sail {

class MatmulTTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& t2, Tensor& out_tensor) {
        launch_arithmetic(t1.dtype, [&](auto pt) {
            // std::cout << decltype(pt)::type << std::endl;
            auto name = decltype(pt)::GetName();

            int M = t1.shape[0];  // ROWS IN A
            int N = t2.shape[1];  // COLS IN B
            int K = t1.shape[1];  // COLS IN A AND ROWS IN B

            // std::cout << (decltype(pt)::GetName() == "float64") << std::endl;

            if (name == "float64") {
                using T = double;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                            1, (T*)t1.data, K, (T*)t2.data, N, 1,
                            (T*)out_tensor.data, N);
                // } else if (name == "float32") {
                //     using T = float;
                //     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M,
                //     N, K,
                //                 1, (T*)t1.data, K, (T*)t2.data, N, 1,
                //                 (T*)out_tensor.data, N); // not sure why this
                //                 doesnt work
            } else {
                using T = typename decltype(pt)::type;
                T* matA = (T*)t1.data;
                T* matB = (T*)t2.data;
                T* matC = (T*)out_tensor.data;
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
};

}  // namespace sail