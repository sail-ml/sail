#pragma once

#include <immintrin.h>
#include <omp.h>
#include <cassert>  // needed for xsimd
#include <cmath>    // needed for xsimd
#include "../../Tensor.h"
#include "../../ops/ops.h"
#include "../base.h"
#include "../elementwise.h"
#include "../unary.h"
#include "xsimd/xsimd.hpp"
namespace sail {

class LogSoftmaxKernel : public Kernel {
   public:
    void execute(Tensor& t1, Tensor& out_tensor, int axis) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;

            Tensor max = ops::max(t1, axis, true);
            Tensor t2 = t1 - max;
            Tensor s = ops::sum(t2, axis, true);
            s = ops::broadcast_to(s, t2.get_shape());
            struct Impl {
                T one = (T)1;
                inline void call_base(T x1, T s_val, T& out) {
                    T ex = (T)std::exp((double)x1);
                    ex = ex / s_val;
                    out = (T)std::log(double(ex));
                }
            };
            BinaryElementwiseNoAvx<T, T, T>(Impl{}, true, t2, s, out_tensor);
        });
    }
};

class SoftmaxMulSumKernel : public Kernel {
   public:
    void execute(Tensor& t1, Tensor& targets, Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            launch_arithmetic(targets.get_dtype(), [&](auto pt2) {
                using DtypeType = decltype(pt);
                using T = typename DtypeType::type;
                using T2 = typename decltype(pt2)::type;

                int i = 0;

                T __restrict__* p1;
                T2* targ;
                T __restrict__* p3;

                p1 = static_cast<T*>(t1.get_data());
                targ = static_cast<T2*>(targets.get_data());
                p3 = static_cast<T*>(out_tensor.get_data());

                int numel = targets.numel();

                int start = 0;
                int size = t1.get_shape().shape[1];
                for (int i = 0; i < numel; i++) {
                    int jump = (int)targ[i];
                    p3[0] += p1[start + jump];
                    start += size;
                }
            });
        });
    }
};

class SoftmaxBackwardSubtractKernel : public Kernel {
   public:
    void execute(Tensor& y, Tensor& targets, Tensor& out_tensor) {
        launch_arithmetic(y.get_dtype(), [&](auto pt) {
            launch_arithmetic(targets.get_dtype(), [&](auto pt2) {
                using DtypeType = decltype(pt);
                using T = typename DtypeType::type;
                using T2 = typename decltype(pt2)::type;

                int i = 0;

                T __restrict__* p1;
                T2* targ;
                T __restrict__* p3;

                p1 = static_cast<T*>(y.get_data());
                targ = static_cast<T2*>(targets.get_data());
                p3 = static_cast<T*>(out_tensor.get_data());

                int size = y.get_shape().shape[1];
                int numel = y.numel();

                int start = (int)targ[0];
                int prev = start;
                int j = 1;
                for (int i = 0; i < numel; i++) {
                    if (i == start) {
                        p3[i] = p1[i] - 1;
                        start += size - prev;
                        start += int(targ[j]);
                        prev = int(targ[j]);
                        j += 1;
                    } else {
                        p3[i] = p1[i];
                    }
                }
            });
        });
    }
};

class TestKernel : public Kernel {
   public:
    void execute(Tensor& t1, Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;
            T* data = (T*)out_tensor.get_data();
#pragma omp parallel for
            for (int i = 0; i < t1.get_shape().shape[0]; i++) {
                int base = t1.get_shape().strides[0] * i;
                Tensor ind = t1[i];
                Tensor max = ops::max(ind);
                // std::cout << "max" << std::endl;
                Tensor y = ind - max;
                // std::cout << "y1" << std::endl;
                y = ops::exp(y);
                // std::cout << "y2" << std::endl;
                Tensor sy = ops::sum(y);
                // std::cout << "sy" << std::endl;
                y = y - ops::log(sy);
                // std::cout << "fin" << std::endl;
                int k = 0;
                for (int j = base; j < base + y.numel(); j++) {
                    data[j] = ((T*)y.get_data())[k];
                    k += 1;
                }
            }
        });
    }
};

}  // namespace sail