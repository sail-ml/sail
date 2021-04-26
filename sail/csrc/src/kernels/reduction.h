#pragma once

#include <immintrin.h>

#include "../Tensor.h"
#include "../factories.h"
#include "base.h"

namespace sail {

/** begin block
 * name = [Add, Sub, Multiply, Divide]
 * baseOp = [+, -, *, /]
 * avxOp = [add, sub, mul, div]
 */

class SumTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, Tensor& out_tensor) {
        launch_arithmetic(t1.storage.dtype, [&](auto pt) {
            // std::cout << decltype(pt)::type << std::endl;
            using T = typename decltype(pt)::type;
            using avx_name = typename decltype(pt)::avx_type;
            struct Impl {
                inline void call_base(T x1, T& out) { out = out + x1; }
            };

            // ElemetwiseAVX<T, Impl, avx_name>(Impl{}, t1, t2, out_tensor);
            UnaryAVX<T, Impl, avx_name>(Impl{}, t1, out_tensor);
        });
    }
};

class MeanTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, Tensor& out_tensor) {
        SumTKernel().execute(t1, out_tensor);
        Tensor lt = empty_scalar(out_tensor.storage.dtype);
        int size = out_tensor.storage.numel();
        lt.storage.data = &size;
        out_tensor = out_tensor / lt;
        lt.free();
    }
};

/** end block **/

}  // namespace sail
