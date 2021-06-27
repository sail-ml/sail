#include "kernels/Binary.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/cpu/Impls.h"
#include "kernels/cpu/loops.h"
#include "kernels/dispatch.h"

namespace sail {

namespace internal {

namespace {
template <typename T, typename avx_type>
struct AddImpl : cpu::BinaryImpl<T, avx_type> {
    inline void call_base(T &x1, T &x2, T &out) { out = x1 + x2; }
    inline avx_type avx_fcn(avx_type &a, avx_type &b) { return a + b; }
};
void add_kernel(const Tensor &t1, const Tensor &t2, Tensor &out,
                bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        using avx_type = xsimd::simd_type<T>;
        static auto imp = AddImpl<T, avx_type>{};

        cpu::BinaryElementwise<T>(imp, broadcast, t1, t2, out);
    });
}

template <typename T, typename avx_type>
struct SubImpl : cpu::BinaryImpl<T, avx_type> {
    inline void call_base(T &x1, T &x2, T &out) { out = x1 - x2; }
    inline avx_type avx_fcn(avx_type &a, avx_type &b) { return a - b; }
};
void sub_kernel(const Tensor &t1, const Tensor &t2, Tensor &out,
                bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        using avx_type = xsimd::simd_type<T>;
        static auto imp = SubImpl<T, avx_type>{};

        cpu::BinaryElementwise<T>(imp, broadcast, t1, t2, out);
    });
}

template <typename T, typename avx_type>
struct MulImpl : cpu::BinaryImpl<T, avx_type> {
    inline void call_base(T &x1, T &x2, T &out) { out = x1 * x2; }
    inline avx_type avx_fcn(avx_type &a, avx_type &b) { return a * b; }
};
void mul_kernel(const Tensor &t1, const Tensor &t2, Tensor &out,
                bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        using avx_type = xsimd::simd_type<T>;
        static auto imp = MulImpl<T, avx_type>{};

        cpu::BinaryElementwise<T>(imp, broadcast, t1, t2, out);
    });
}

template <typename T, typename avx_type>
struct DivImpl : cpu::BinaryImpl<T, avx_type> {
    inline void call_base(T &x1, T &x2, T &out) { out = x1 / x2; }
    inline avx_type avx_fcn(avx_type &a, avx_type &b) { return a / b; }
};
void div_kernel(const Tensor &t1, const Tensor &t2, Tensor &out,
                bool broadcast) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;

        using avx_type = xsimd::simd_type<T>;
        static auto imp = DivImpl<T, avx_type>{};

        cpu::BinaryElementwise<T>(imp, broadcast, t1, t2, out);
    });
}

}  // namespace
REGISTER_AVX_DISPATCH(add_stub, &add_kernel);
REGISTER_AVX_DISPATCH(subtract_stub, &sub_kernel);
// REGISTER_AVX_DISPATCH(divide_stub, &div_kernel);
REGISTER_AVX_DISPATCH(multiply_stub, &mul_kernel);

}  // namespace internal

}  // namespace sail