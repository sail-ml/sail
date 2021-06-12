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
Tensor tensordot(const Tensor& t1, const Tensor& t2, int axes);

Tensor addmm(const Tensor& m1, const Tensor& m2, const Tensor& add);

}  // namespace ops

}  // namespace sail
