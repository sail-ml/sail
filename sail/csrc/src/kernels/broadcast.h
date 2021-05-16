

#pragma once

#include <immintrin.h>

#include "../Tensor.h"
#include "../dtypes.h"
#include "../errors.h"
#include "../factories.h"

namespace sail {

class BroadcastToKernel : public Kernel {
   public:
    void execute(const Tensor& t1, const TensorSize broadcast_shape,
                 Tensor& out_tensor) {
        launch_arithmetic(t1.get_dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;

            TensorSize shape = t1.shape;
            // TensorSize  = out_tensor.shape;
            // assume broadcast_shape has more dimensions because it kinda has
            // to. Cant broadcast down
            int o_i = shape.size() - 1;
            for (int i = broadcast_shape.size() - 1; i >= 0; i--) {
                if (o_i < 0) {
                    continue;
                }
                if ((broadcast_shape[i] != shape[o_i]) && (shape[o_i] != 1)) {
                    throw SailCError(
                        "dimensions cannot be broadcasted together");
                }
                o_i -= 1;
            }

            // using avx_name = typename decltype(pt)::avx_type;
            // struct Impl {
            //     inline void call_base(T x1, T& out) { out = out + x1; }
            // };

            // T* input_data = (T*)t1.get_data();
            // T* output_data = (T*)out_tensor.get_data();

            // if (axis != -1) {
            //     int ms = t1.shape[axis];
            //     int red_jump = 1;
            //     int c = 0;
            //     for (int s : t1.shape) {
            //         if (c > axis) {
            //             red_jump *= s;  //(s * GetDtypeSize(t1.get_dtype()));
            //         }
            //         c += 1;
            //     }

            //     int idx = 0;
            //     int insert_idx = 0;
            //     int inner_count = 0;
            //     T inner_sum = 0;
            //     // size_t size = getTotalSize(t1);
            //     // std::cout << size << std::endl;
            //     while (idx < t1.numel()) {
            //         inner_sum = 0;
            //         for (int i = 0; i < ms; i++) {
            //             inner_sum += input_data[idx + (red_jump * i)];
            //         }
            //         output_data[insert_idx] = inner_sum;
            //         inner_count += 1;
            //         insert_idx += 1;
            //         idx += 1;

            //         if (inner_count == red_jump) {
            //             idx += (red_jump * (ms - 1));
            //             inner_count = 0;
            //         }
            //     }
            // } else {
            //     Unary<T, T>(Impl{}, t1, out_tensor);
            // }
        });
    }
};

}  // namespace sail
