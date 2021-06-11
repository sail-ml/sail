#pragma once
#define OMP_MIN_VALUE \
    512  // 65536/16 // should probably find a real number ot use

#define JUMP_LOOP(numel, jump) for (; i < numel; i += jump)

#include <immintrin.h>
#include <omp.h>

#include "Tensor.h"
#include "utils.h"

using Tensor = sail::Tensor;
namespace sail {

template <typename T, typename Op, typename avx_name>
void ElemetwiseAVX(Op op, const Tensor &arr1, const Tensor &arr2,
                   const Tensor &arr3) {
    T __restrict__ *p1 = static_cast<T *>(arr1.get_data());
    T __restrict__ *p2 = static_cast<T *>(arr2.get_data());
    T __restrict__ *p3 = static_cast<T *>(arr3.get_data());

    int numel = arr1.numel();
    bool aligned = isAlignedAs(p1, arr1.get_info().alignment) &&
                   isAlignedAs(p2, arr1.get_info().alignment);
    int jump = arr1.get_info().jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;

#ifdef S_USE_AVX2
    if (omp) {
        if (aligned) {
#pragma omp parallel for
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_aligned(&p1[i], &p2[i], &p3[i]);
            }
        } else {
#pragma omp parallel for
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_non_aligned(&p1[i], &p2[i], &p3[i]);
            }
        }
    } else {
        if (aligned) {
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_aligned(&p1[i], &p2[i], &p3[i]);
            }
        } else {
            for (int i = 0; i < numel; i += jump) {
                op.call_avx_non_aligned(&p1[i], &p2[i], &p3[i]);
            }
        }
    }

#else
    if (omp) {
#pragma omp parallel for
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[i], p3[i]);
        }

    } else {
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[i], p3[i]);
        }
    }
#endif

    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // std::cout << duration.count() << std::endl;
}
template <typename T, typename Op, typename avx_name>
void ElemetwiseScalarAVX(Op op, const Tensor &arr1, const Tensor &arr2,
                         const Tensor &arr3) {
    T __restrict__ *p1 = static_cast<T *>(arr1.get_data());
    T __restrict__ *p2 = static_cast<T *>(arr2.get_data());
    T __restrict__ *p3 = static_cast<T *>(arr3.get_data());

    int numel = arr1.numel();
    bool aligned = isAlignedAs(p1, arr1.get_info().alignment);
    int jump = arr1.get_info().jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;
    // if (omp)
    // auto start = high_resolution_clock::now();

    // #ifdef USE_AVX2
    //     if (omp) {
    //         if (aligned) {
    // #pragma omp parallel for
    //             for (int i = 0; i < numel; i += jump) {
    //                 avx_name a = _mm256_load_pd(&p1[i]);
    //                 avx_name b = _mm256_load_pd(&p2[0]);
    //                 op.call_avx_aligned(a, b, &p3[i]);
    //             }
    //         } else {
    // #pragma omp parallel for
    //             for (int i = 0; i < numel; i += jump) {
    //                 avx_name a = _mm256_loadu_pd(&p1[i]);
    //                 avx_name b = _mm256_loadu_pd(&p2[0]);
    //                 op.call_avx_non_aligned(a, b, &p3[i]);
    //             }
    //         }
    //     } else {
    //         if (aligned) {
    //             for (int i = 0; i < numel; i += jump) {
    //                 avx_name a = _mm256_load_pd(&p1[i]);
    //                 avx_name b = _mm256_load_pd(&p2[0]);
    //                 op.call_avx_aligned(a, b, &p3[i]);
    //             }
    //         } else {
    //             for (int i = 0; i < numel; i += jump) {
    //                 avx_name a = _mm256_loadu_pd(&p1[i]);
    //                 avx_name b = _mm256_loadu_pd(&p2[0]);
    //                 op.call_avx_non_aligned(a, b, &p3[i]);
    //             }
    //         }
    //     }

    // #else
    if (omp) {
#pragma omp parallel for
        for (i = 0; i < numel; i += 1) {
            // std::cout << p1[i] << ", " << p2[0] << std::endl;
            op.call_base(p1[i], p2[0], p3[i]);
        }

    } else {
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[0], p3[i]);
        }
    }
    // #endif
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // std::cout << duration.count() << std::endl;
}

template <typename T, typename T_out, typename Op>
void Elemetwise(Op op, const Tensor &arr1, const Tensor &arr2) {
    T *p1 = static_cast<T *>(arr1.get_data());
    T_out *p2 = static_cast<T_out *>(arr2.get_data());

    int numel = arr1.numel();
    bool aligned = isAlignedAs(p1, arr1.get_info().alignment) &&
                   isAlignedAs(p2, arr1.get_info().alignment);
    int jump = arr1.get_info().jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;

    if (omp) {
#pragma omp parallel for
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[i]);
        }

    } else {
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], p2[i]);
        }
    }
}

template <typename T, typename Op, typename avx_name>
void UnaryAVX(Op op, const Tensor &arr1, const Tensor &arr_out) {
    T __restrict__ *p1 = static_cast<T *>(arr1.get_data());
    T __restrict__ *p_out = static_cast<T *>(arr_out.get_data());

    int numel = arr1.numel();
    bool aligned = isAlignedAs(p1, arr1.get_info().alignment);
    int jump = arr1.get_info().jump;
    int i = 0;
    bool omp = numel >= OMP_MIN_VALUE;
    T sum = 0;

    // if (omp)
    // #ifdef USE_AVX2
    //     if (omp) {
    //         if (aligned) {
    //             #pragma omp parallel for
    //             for(int i = 0; i < numel; i+=jump) {
    //                 avx_name a = _mm256_load_pd(&p1[i]);
    //                 avx_name b = _mm256_load_pd(*p_out);
    //                 op.call_avx_aligned(a, b, *p_out);
    //             }
    //         } else {
    //             #pragma omp parallel for
    //             for(int i = 0; i < numel; i+=jump) {
    //                 avx_name a = _mm256_loadu_pd(&p1[i]);
    //                 op.call_avx_non_aligned(a, *p_out);
    //             }
    //         }
    //     } else {
    //        if (aligned) {
    //             for(int i = 0; i < numel; i+=jump) {
    //                 avx_name a = _mm256_load_pd(&p1[i]);
    //                 op.call_avx_aligned(a, *p_out);
    //             }
    //         } else {
    //             for(int i = 0; i < numel; i+=jump) {
    //                 avx_name a = _mm256_loadu_pd(&p1[i]);
    //                 op.call_avx_non_aligned(a, *p_out);
    //             }
    //         }
    //     }

    // #else
    if (omp) {
#pragma omp parallel for reduction(+ : sum)
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], sum);
        }

    } else {
        for (i = 0; i < numel; i += 1) {
            op.call_base(p1[i], sum);
        }
    }
    *p_out = sum;
    // #endif
}

// // template <typename T, typename Op>
// // void Elemetwise(Op op, const Tensor &arr1, const Tensor &arr2, const
// Tensor &arr3) {
// //     T *p1 = static_cast<T*>(arr1.storage.get_data());
// //     T *p2 = static_cast<T*>(arr2.storage.get_data());
// //     T *p3 = static_cast<T*>(arr3.storage.get_data());

// //     int numel = arr1.storage.numel();
// //     int i;
// //     if (numel > OMP_MIN_VALUE)
// //         #pragma omp parallel for
// //     for (i = 0; i < numel; i += 1) {
// //         op.call_base(p1[i], p2[i], p3[i]);
// //     }
// // }

class Kernel {
   public:
    Kernel() = default;

    virtual ~Kernel() = default;

    Kernel(const Kernel &) = delete;
    Kernel(Kernel &&) = delete;
    Kernel &operator=(const Kernel &) = delete;
    Kernel &operator=(Kernel &&) = delete;
};

}  // namespace sail
