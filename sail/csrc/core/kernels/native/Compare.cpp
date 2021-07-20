// allow-no-header

#include "kernels/Compare.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"

namespace sail {

namespace internal {

namespace {

void clip_min_kernel(const Tensor& t1, const double min, Tensor& out) {
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            T min;
            Impl(T min) : min(min){};
            inline void call_base(T x1, T& out) {
                if (x1 < min) {
                    out = min;
                    return;
                }
                out = x1;
            }
        };
        native::UnaryElementwise<T>(Impl{(T)min}, t1, out);
    });
}

void clip_max_kernel(const Tensor& t1, const double max, Tensor& out) {
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            T max;
            Impl(T max) : max(max){};
            inline void call_base(T x1, T& out) {
                if (x1 > max) {
                    out = max;
                    return;
                }
                out = x1;
            }
        };
        native::UnaryElementwise<T>(Impl{(T)max}, t1, out);
    });
}

void clip_kernel(const Tensor& t1, const double min, const double max,
                 Tensor& out) {
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;
        struct Impl {
            T min, max;
            Impl(T min, T max) : min(min), max(max){};
            inline void call_base(T x1, T& out) {
                out = std::clamp(x1, min, max);
            }
        };
        native::UnaryElementwise<T>(Impl{(T)min, (T)max}, t1, out);
    });
}

}  // namespace
REGISTER_ARCH_DISPATCH(clip_min_stub, DEFAULT, &clip_min_kernel);
REGISTER_ARCH_DISPATCH(clip_max_stub, DEFAULT, &clip_max_kernel);
REGISTER_ARCH_DISPATCH(clip_stub, DEFAULT, &clip_kernel);

namespace {

void check_type(const Tensor& t) {
    SAIL_CHECK(t.get_dtype() == Dtype::sBool,
               "Comparison operators must write to a boolean");
}

template <typename T, typename Op>
void native_compare_launch_binary_elementwise(Op op, const Tensor& t1,
                                              const Tensor& t2,
                                              const Tensor& out) {
    int i = 0;
    const int jump = t1.get_info().jump;

    T* p1;
    T* p2;
    bool* p3;

    p1 = static_cast<T*>(t1.get_data());
    p2 = static_cast<T*>(t2.get_data());
    p3 = static_cast<bool*>(out.get_data());

    TensorShape s1 = t1.get_shape();
    TensorShape s2 = t2.get_shape();

    MultiTensorIterator iter = MultiTensorIterator(s1).add_input(s2);

    int inner_loop_size = iter.inner_loop_size();
    int outer_steps = iter.out_loop_size();

    int z = 0;
    for (int i = 0; i < outer_steps; i++) {
        for (int j = 0; j < inner_loop_size; j += 1) {
            op.call_base(p1[iter.d_ptrs[0]], p2[iter.d_ptrs[1]], p3[z]);
            iter.advance_d_ptr(1);
            z += 1;
        }
        iter.backup_d_ptr();
        iter.next();
    }
}

void equal_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
                  bool broadcast) {
    check_type(out_tensor);
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, bool& out) {
                if (x1 == x2) {
                    out = true;
                } else {
                    out = false;
                }
            }
        };

        native_compare_launch_binary_elementwise<T>(Impl{}, t1, t2, out_tensor);
    });
}
void lt_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
               bool broadcast) {
    check_type(out_tensor);
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, bool& out) {
                if (x1 < x2) {
                    out = true;
                } else {
                    out = false;
                }
            }
        };

        native_compare_launch_binary_elementwise<T>(Impl{}, t1, t2, out_tensor);
    });
}
void gt_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
               bool broadcast) {
    check_type(out_tensor);
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, bool& out) {
                if (x1 > x2) {
                    out = true;
                } else {
                    out = false;
                }
            }
        };

        native_compare_launch_binary_elementwise<T>(Impl{}, t1, t2, out_tensor);
    });
}
void lte_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
                bool broadcast) {
    check_type(out_tensor);
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, bool& out) {
                if (x1 <= x2) {
                    out = true;
                } else {
                    out = false;
                }
            }
        };

        native_compare_launch_binary_elementwise<T>(Impl{}, t1, t2, out_tensor);
    });
}
void gte_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
                bool broadcast) {
    check_type(out_tensor);
    dispatch_all_numeric_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, bool& out) {
                if (x1 > x2) {
                    out = true;
                } else {
                    out = false;
                }
            }
        };

        native_compare_launch_binary_elementwise<T>(Impl{}, t1, t2, out_tensor);
    });
}
}  // namespace

REGISTER_ONLY_NATIVE_DISPATCH(equal_stub, &equal_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(lt_stub, &lt_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(gt_stub, &gt_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(lte_stub, &lte_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(gte_stub, &gte_kernel);

}  // namespace internal

}  // namespace sail