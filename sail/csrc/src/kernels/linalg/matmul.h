#pragma once
#define OMP_MIN_VALUE 512  // should probably find a real number ot use

#include <cblas.h>
#include <immintrin.h>
#include <iostream>

#include "../../Tensor.h"
#include "../base.h"
#include "../unary.h"

namespace sail {

class DotTTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            // std::cout << decltype(pt)::type << std::endl;
            auto name = decltype(pt)::GetName();

            int M = t1.get_shape().shape[0];  // ROWS IN A
            int K = t1.get_shape().shape[1];  // COLS IN A AND ROWS IN B
            int N = t2.get_shape().shape[1];  // COLS IN B

            // std::cout << (decltype(pt)::GetName() == "float64") << std::endl;

            // if (name == "float64") {
            //     using T = double;
            //     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N,
            //     K,
            //                 1, (T*)t1.get_data(), K, (T*)t2.get_data(), N, 1,
            //                 (T*)out_tensor.get_data(), N);
            //     // } else if (name == "float32") {
            //     //     using T = float;
            //     //     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            //     // M,
            //     //     N, K,
            //     //                 1, (T*)t1.get_data(), K,
            //     // (T*)t2.get_data(),
            //     //                 N, 1, (T*)out_tensor.get_data(), N); //
            //     // not
            //     //                 sure why this doesnt work
            // } else {
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
            // }
        });
    }
};  // namespace sail

}  // namespace sail