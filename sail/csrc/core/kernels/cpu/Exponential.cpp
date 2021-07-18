#include "Tensor.h"
#include "constants.h"
#include "dtypes.h"
#include "exception.h"
#include "kernels/Binary.h"
#include "kernels/cpu/Impls.h"
#include "kernels/cpu/loops.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "xsimd/xsimd.hpp"

namespace sail {

namespace internal {

template <typename T, typename avx_type>
struct LogImpl : cpu::UnaryImpl<T, avx_type> {
    inline void call_base(T& x1, T& out) override {
        out = (T)std::log((double)x1);
    }
    inline avx_type avx_fcn(avx_type& a) override { return xsimd::log(a); }
};
template <typename T>
struct LogImplNoFP {
    inline void call_base(T& x1, T& out) { out = (T)std::log((double)x1); }
};
namespace {
void log_kernel(const Tensor& t1, Tensor& out) {
    dispatch_fp_int_types(t1.get_dtype(),
                          [&](auto pt) {
                              using DtypeType = decltype(pt);
                              using T = typename DtypeType::type;
                              using avx_type = xsimd::simd_type<T>;

                              static auto imp = LogImpl<T, avx_type>{};

                              cpu::UnaryElementwise<T>(imp, t1.is_view(), t1,
                                                       out);
                              return;
                          },
                          [&](auto pt2) {
                              using DtypeType = decltype(pt2);
                              using T = typename DtypeType::type;

                              static auto imp = LogImplNoFP<T>{};
                              native::UnaryElementwise<T>(imp, t1, out);
                              return;
                          });
}

}  // namespace
REGISTER_AVX_DISPATCH(log_stub, &log_kernel);

}  // namespace internal

}  // namespace sail