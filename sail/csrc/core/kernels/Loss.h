#pragma once

#include <iostream>
#include "Tensor.h"
#include "kernels/dispatch.h"

#define TRANS "T"
#define NO_TRANS "N"
#define CONJ_TRANS "C"

namespace sail {

namespace internal {

using mse_loss = void (*)(const Tensor& t1, const Tensor& t2,
                          Tensor& out_tensor);

DECLARE_DISPATCH(mse_loss, mse_stub);

}  // namespace internal

}  // namespace sail