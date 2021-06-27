#include "kernels/Compare.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"

namespace sail {

namespace internal {

namespace {

void clip_min_kernel(const Tensor& t1, const double min, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
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
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
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
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;
        struct Impl {
            T min, max;
            Impl(T min, T max) : min(min), max(max){};
            inline void call_base(T x1, T& out) {
                if (x1 > max) {
                    out = max;
                    return;
                } else if (x1 < min) {
                    out = min;
                    return;
                }
                out = x1;
            }
        };
        native::UnaryElementwise<T>(Impl{(T)min, (T)max}, t1, out);
    });
}

}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(clip_min_stub, &clip_min_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(clip_max_stub, &clip_max_kernel);
REGISTER_ONLY_NATIVE_DISPATCH(clip_stub, &clip_kernel);

namespace {
void equal_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
                  bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, T& out) {
                if (x1 == x2) {
                    out = (T)1;
                } else {
                    out = (T)0;
                }
            }
        };

        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out_tensor);
    });
}
void lt_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
               bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, T& out) {
                if (x1 < x2) {
                    out = (T)1;
                } else {
                    out = (T)0;
                }
            }
        };

        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out_tensor);
    });
}
void gt_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
               bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, T& out) {
                if (x1 > x2) {
                    out = (T)1;
                } else {
                    out = (T)0;
                }
            }
        };

        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out_tensor);
    });
}
void lte_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
                bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, T& out) {
                if (x1 <= x2) {
                    out = (T)1;
                } else {
                    out = (T)0;
                }
            }
        };

        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out_tensor);
    });
}
void gte_kernel(const Tensor& t1, const Tensor& t2, const Tensor& out_tensor,
                bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        struct Impl {
            inline void call_base(T& x1, T& x2, T& out) {
                if (x1 > x2) {
                    out = (T)1;
                } else {
                    out = (T)0;
                }
            }
        };

        native::BinaryElementwise<T>(Impl{}, broadcast, t1, t2, out_tensor);
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