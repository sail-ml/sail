#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cassert>  // needed for xsimd
#include <vector>
#include "Tensor.h"
#include "dtypes.h"
#include "error.h"
#include "kernels/kernel_utils.h"
#include "tensor_iterator.h"
#include "tensor_shape.h"
#include "xsimd/xsimd.hpp"

namespace sail {

namespace cpu {

template <typename T, typename avx_type>
struct BinaryImpl {
    int size = avx_type::size;
    avx_type scal;
    virtual inline void call_base(T &x1, T &x2, T &out) {
        THROW_ERROR_DETAILED(SailCError, "No");
    }
    virtual inline avx_type avx_fcn(avx_type &a, avx_type &b) {
        THROW_ERROR_DETAILED(SailCError, "No");
    }

    inline void set_scalar_val(T &x1) { scal = avx_type::broadcast(x1); }
    inline void call_avx_aligned(T *x1, T *x2, T *out) {
        avx_type a = xsimd::load_aligned(x1);
        avx_type b = xsimd::load_aligned(x2);
        auto c = avx_fcn(a, b);
        xsimd::store_aligned(out, c);
    }

    inline void iterator_avx_second(T *x2, T *out) {
        avx_type b = xsimd::load_unaligned(x2);
        auto c = avx_fcn(b, scal);
        c.store_unaligned(out);
    }
    inline void iterator_avx_first(T *x2, T *out) {
        avx_type b = xsimd::load_unaligned(x2);
        auto c = avx_fcn(scal, b);
        c.store_unaligned(out);
    }

    inline void call_avx_non_aligned(T *x1, T *x2, T *out) {
        avx_type a = xsimd::load_unaligned(x1);
        avx_type b = xsimd::load_unaligned(x2);
        auto c = avx_fcn(a, b);
        xsimd::store_unaligned(out, c);
    }
};

template <typename T, typename avx_type>
struct UnaryImpl {
    int size = avx_type::size;
    avx_type scal;
    virtual inline void call_base(T &x1, T &out) {
        THROW_ERROR_DETAILED(SailCError, "No");
    }
    virtual inline avx_type avx_fcn(avx_type &a) {
        THROW_ERROR_DETAILED(SailCError, "No");
    }

    inline void set_scalar_val(T &x1) { scal = avx_type::broadcast(x1); }
    inline void call_avx_aligned(T *x1, T *out) {
        avx_type a = xsimd::load_aligned(x1);
        auto c = avx_fcn(a);
        xsimd::store_aligned(out, c);
    }

    inline void call_avx_non_aligned(T *x1, T *out) {
        avx_type a = xsimd::load_unaligned(x1);
        auto c = avx_fcn(a);
        xsimd::store_unaligned(out, c);
    }
    inline void call_avx_scalar(T *out) {
        auto c = avx_fcn(scal);
        xsimd::store_unaligned(out, c);
    }
};

}  // namespace cpu

}  // namespace sail