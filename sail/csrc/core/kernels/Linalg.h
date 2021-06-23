#pragma once

#include <iostream>
#include "Tensor.h"
#include "kernels/dispatch.h"

#define TRANS "T"
#define NO_TRANS "N"
#define CONJ_TRANS "C"

namespace sail {

namespace internal {

using matmul_fcn = void (*)(const Tensor& t1, const Tensor& t2,
                            Tensor& out_tensor, bool empty, std::string trans_a,
                            std::string trans_b);

DECLARE_DISPATCH(matmul_fcn, matmul_stub);

}  // namespace internal

}  // namespace sail