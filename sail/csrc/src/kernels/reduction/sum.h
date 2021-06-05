#pragma once

#include <immintrin.h>

#include "../../Tensor.h"
#include "../../dtypes.h"
#include "../../factories.h"
#include "../reduction.h"

namespace sail {

class SumTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, Tensor& out_tensor, int axis) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            // std::cout << decltype(pt)::type << std::endl;
            using T = typename decltype(pt)::type;
            using avx_name = typename decltype(pt)::avx_type;
            struct Impl {
                inline void call_base(T x1, T& out) { out = out + x1; }
            };

            T* input_data = (T*)t1.get_data();
            T* output_data = (T*)out_tensor.get_data();

            if (axis != -1) {
                Reduction<T, T>(Impl{}, t1, out_tensor, axis);
            } else {
                Reduction<T, T>(Impl{}, t1, out_tensor);
            }
        });
    }
};

// class MeanTKernel : public Kernel {
//    public:
//     void execute(const Tensor& t1, Tensor& out_tensor) {
//         SumTKernel().execute(t1, out_tensor, -1);  // need to change
//         Tensor lt = empty_scalar(out_tensor.get_dtype());
//         int size = out_tensor.numel();
//         lt.get_data() = std::make_shared<void>((void*)&size);
//         out_tensor = out_tensor / lt;
//         lt.free();
//     }
// };

}  // namespace sail
