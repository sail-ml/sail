#pragma once

#include <iostream>

#include "Tensor.h"
#include "dtypes.h"
#include "kernels/kernel.h"

#define TRANS "T"
#define NO_TRANS "N"
#define CONJ_TRANS "C"

namespace sail {

namespace ops {

Tensor matmul(const Tensor& t1, const Tensor& t2,
              std::string trans_a = NO_TRANS, std::string trans_b = NO_TRANS);
Tensor tensordot(const Tensor& t1, const Tensor& t2, LongVec t1_dim,
                 LongVec t2_dim);
inline Tensor tensordot(const Tensor& t1, const Tensor& t2, int axes) {
    LongVec t1_dim, t2_dim;

    while (axes--) {
        t1_dim.push_back(axes);
        t2_dim.push_back(axes);
    }
    std::reverse(t1_dim.begin(), t1_dim.end());
    std::reverse(t2_dim.begin(), t2_dim.end());

    return tensordot(t1, t2, t1_dim, t2_dim);
}

Tensor addmm(const Tensor& m1, const Tensor& m2, const Tensor& add);

}  // namespace ops

}  // namespace sail
