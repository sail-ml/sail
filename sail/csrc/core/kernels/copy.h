#pragma once
#define OMP_MIN_VALUE 512  // should probably find a real number ot use

#include <immintrin.h>
#include <iostream>

#include "Tensor.h"
#include "base.h"
#include "unary.h"

namespace sail {

class CopyTTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            launch_arithmetic(out_tensor.get_dtype(), [&](auto xt) {
                using T_in = typename decltype(pt)::type;
                using T_out = typename decltype(xt)::type;

                struct Impl {
                    inline void call_base(T_in x1, T_out& out) {
                        out = static_cast<T_out>(x1);
                    }
                };

                Unary<T_in, T_out>(Impl{}, t1, out_tensor);
            });
        });
    }
};

}  // namespace sail