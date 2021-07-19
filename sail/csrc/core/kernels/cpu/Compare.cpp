// allow-no-header

#include "kernels/Compare.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/cpu/Impls.h"
#include "kernels/cpu/loops.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"

namespace sail {

namespace internal {

namespace {

template <typename T, typename avx_type>
struct ClipMinImpl : cpu::UnaryImpl<T, avx_type> {
    T min_base;
    avx_type min_avx;
    ClipMinImpl(T min) : min_base(min), min_avx(avx_type::broadcast(min)){};
    inline void call_base(T& x1, T& out) override {
        out = std::max(x1, min_base);
    }
    inline avx_type avx_fcn(avx_type& a) override {
        return xsimd::max(a, min_avx);
    }
};
void clip_min_kernel(const Tensor& t1, const double min, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;
        using avx_type = xsimd::simd_type<T>;

        static auto imp = ClipMinImpl<T, avx_type>{(T)min};

        cpu::UnaryElementwise<T>(imp, t1.is_view(), t1, out);
    });
}

template <typename T, typename avx_type>
struct ClipMaxImpl : cpu::UnaryImpl<T, avx_type> {
    ClipMaxImpl(T max) : max_base(max), max_avx(avx_type::broadcast(max)){};
    T max_base;
    avx_type max_avx;
    inline void call_base(T& x1, T& out) override {
        out = std::min(x1, max_base);
    }
    inline avx_type avx_fcn(avx_type& a) override {
        return xsimd::min(a, max_avx);
    }
};
void clip_max_kernel(const Tensor& t1, const double max, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;
        using avx_type = xsimd::simd_type<T>;

        static auto imp = ClipMaxImpl<T, avx_type>{(T)max};

        cpu::UnaryElementwise<T>(imp, t1.is_view(), t1, out);
    });
}

template <typename T, typename avx_type>
struct ClipImpl : cpu::UnaryImpl<T, avx_type> {
    T max_base;
    T min_base;
    avx_type max_avx;
    avx_type min_avx;
    ClipImpl(T min, T max)
        : min_base(min),
          max_base(max),
          min_avx(avx_type::broadcast(min)),
          max_avx(avx_type::broadcast(max)){};
    inline void call_base(T& x1, T& out) override {
        out = std::clamp(x1, min_base, max_base);
    }
    inline avx_type avx_fcn(avx_type& a) override {
        return xsimd::clip(a, min_avx, max_avx);
    }
};

void clip_kernel(const Tensor& t1, const double min, const double max,
                 Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        dispatch_all_types(t1.get_dtype(), [&](auto pt) {
            using DtypeType = decltype(pt);
            using T = typename DtypeType::type;
            using avx_type = xsimd::simd_type<T>;

            static auto imp = ClipImpl<T, avx_type>{(T)min, (T)max};

            cpu::UnaryElementwise<T>(imp, t1.is_view(), t1, out);
        });
    });
}

}  // namespace
REGISTER_AVX_DISPATCH(clip_min_stub, &clip_min_kernel);
REGISTER_AVX_DISPATCH(clip_max_stub, &clip_max_kernel);
REGISTER_AVX_DISPATCH(clip_stub, &clip_kernel);

}  // namespace internal

}  // namespace sail