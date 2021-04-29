#pragma once

#include <immintrin.h>

#include "../Tensor.h"
#include "../dtypes.h"
#include "../factories.h"
#include "unary.h"

namespace sail {

inline size_t getTotalSize(const Tensor& t1) {
    auto size = GetDtypeSize(t1.dtype);
    for (size_t value : t1.shape) {
        size = size * value;
    }
    return size;
}

class SumTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, Tensor& out_tensor, int axis) {
        launch_arithmetic(t1.dtype, [&](auto pt) {
            // std::cout << decltype(pt)::type << std::endl;
            using T = typename decltype(pt)::type;
            using avx_name = typename decltype(pt)::avx_type;
            struct Impl {
                inline void call_base(T x1, T& out) { out = out + x1; }
            };

            T* input_data = (T*)t1.data;
            T* output_data = (T*)out_tensor.data;

            if (axis != -1) {
                int red_jump = 1;
                int c = 0;
                for (int s : t1.shape) {
                    if (c > axis) {
                        red_jump *= s;  //(s * GetDtypeSize(t1.dtype));
                    }
                    c += 1;
                }

                int idx = 0;
                int insert_idx = 0;
                int inner_count = 0;
                while (idx < getTotalSize(t1)) {
                    output_data[insert_idx] =
                        input_data[idx] + input_data[idx + red_jump];
                    inner_count += 1;
                    insert_idx += 1;
                    idx += 1;

                    if (inner_count == red_jump) {
                        idx += red_jump;
                        inner_count = 0;
                    }
                }
            } else {
                Unary<T, T>(Impl{}, t1, out_tensor);
            }
        });
    }
};

class MeanTKernel : public Kernel {
   public:
    void execute(const Tensor& t1, Tensor& out_tensor) {
        SumTKernel().execute(t1, out_tensor, -1);  // need to change
        Tensor lt = empty_scalar(out_tensor.dtype);
        int size = out_tensor.numel();
        lt.data = &size;
        out_tensor = out_tensor / lt;
        lt.free();
    }
};

}  // namespace sail
