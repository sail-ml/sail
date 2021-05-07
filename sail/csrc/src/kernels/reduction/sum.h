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
                int ms = t1.shape_details.shape[axis];
                int red_jump = 1;
                int c = 0;
                for (int s : t1.shape_details.shape) {
                    if (c > axis) {
                        red_jump *= s;  //(s * GetDtypeSize(t1.dtype));
                    }
                    c += 1;
                }

                int idx = 0;
                int insert_idx = 0;
                int inner_count = 0;
                T inner_sum = 0;
                // size_t size = getTotalSize(t1);
                // std::cout << size << std::endl;
                while (idx < t1.numel()) {
                    inner_sum = 0;
                    for (int i = 0; i < ms; i++) {
                        inner_sum += input_data[idx + (red_jump * i)];
                    }
                    output_data[insert_idx] = inner_sum;
                    inner_count += 1;
                    insert_idx += 1;
                    idx += 1;

                    if (inner_count == red_jump) {
                        idx += (red_jump * (ms - 1));
                        inner_count = 0;
                    }
                }
            } else {
                Reduction<T, T>(Impl{}, t1, out_tensor);
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
