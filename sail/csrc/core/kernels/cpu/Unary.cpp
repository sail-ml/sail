// allow-no-header

#include "kernels/Unary.h"
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
struct NegateImpl : cpu::UnaryImpl<T, avx_type> {
    inline void call_base(T& x1, T& out) override { out = -x1; }
    inline avx_type avx_fcn(avx_type& a) override { return -a; }
};
void negate_kernel(const Tensor& t1, Tensor& out) {
    dispatch_all_types(t1.get_dtype(), [&](auto pt) {
        using DtypeType = decltype(pt);
        using T = typename DtypeType::type;
        using avx_type = xsimd::simd_type<T>;

        static auto imp = NegateImpl<T, avx_type>{};

        cpu::UnaryElementwise<T>(imp, t1.is_view(), t1, out);
        return;
    });
}

}  // namespace
REGISTER_AVX_DISPATCH(negate_stub, &negate_kernel);

}  // namespace internal

}  // namespace sail