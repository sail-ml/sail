// allow-impl-in-header allow-no-source
#pragma once

#include <immintrin.h>
#include <cstring>
#include <iostream>

#include "exception.h"
#include "utils.h"

enum class Dtype {
    sBool = 1,
    sUInt8,
    sInt8,

    sUInt16,
    sInt16,

    sUInt32,
    sInt32,

    sUInt64,
    sInt64,

    sFloat32,
    sFloat64,
};

inline std::ostream& operator<<(std::ostream& os, const Dtype& dt) {
    switch (dt) {
        case Dtype::sBool:
            os << "bool";
            break;
        case Dtype::sInt8:
            os << "int8";
            break;
        case Dtype::sUInt8:
            os << "uint8";
            break;
        case Dtype::sInt16:
            os << "int16";
            break;
        case Dtype::sUInt16:
            os << "uint16";
            break;
        case Dtype::sInt32:
            os << "int32";
            break;
        case Dtype::sUInt32:
            os << "uint32";
            break;
        case Dtype::sInt64:
            os << "int64";
            break;
        case Dtype::sUInt64:
            os << "uint64";
            break;
        case Dtype::sFloat32:
            os << "float32";
            break;
        default:
            os << "float64";
            break;
    }
    return os;
}

inline Dtype default_dtype = Dtype::sFloat32;

enum class Dtypekind {
    sBool = 0,
    sInt,
    sUInt,
    sFloat,
};

template <typename T>
struct PrimitiveType;

template <typename T>
struct NonPrimitveType;

#define DEFINE_PRIMITIVE_TYPE(name, code, dtype, next, kind, t, dst, avx) \
    template <>                                                           \
    struct PrimitiveType<t> {                                             \
        using type = t;                                                   \
        using device_storage_type = dst;                                  \
        using avx_type = avx;                                             \
        static constexpr char sCharCode = code;                           \
        static constexpr Dtype sDtype = dtype;                            \
        static constexpr Dtype nextDtype = next;                          \
        static constexpr int64_t sElementSize = sizeof(type);             \
        static constexpr Dtypekind sKind = kind;                          \
        static const char* GetName() { return name; }                     \
    }

DEFINE_PRIMITIVE_TYPE("bool", 'b', Dtype::sBool, Dtype::sBool, Dtypekind::sBool,
                      bool, bool, __m256i);

DEFINE_PRIMITIVE_TYPE("uint8", 'u', Dtype::sUInt8, Dtype::sInt16,
                      Dtypekind::sUInt, uint8_t, uint8_t, __m256i);
DEFINE_PRIMITIVE_TYPE("int8", 'i', Dtype::sInt8, Dtype::sInt16, Dtypekind::sInt,
                      int8_t, int8_t, __m256i);

DEFINE_PRIMITIVE_TYPE("uint16", 'u', Dtype::sUInt16, Dtype::sInt32,
                      Dtypekind::sUInt, uint16_t, uint16_t, __m256i);
DEFINE_PRIMITIVE_TYPE("int16", 'i', Dtype::sInt16, Dtype::sInt32,
                      Dtypekind::sInt, int16_t, int16_t, __m256i);

DEFINE_PRIMITIVE_TYPE("uint32", 'u', Dtype::sUInt32, Dtype::sInt64,
                      Dtypekind::sUInt, uint32_t, uint32_t, __m256i);
DEFINE_PRIMITIVE_TYPE("int32", 'i', Dtype::sInt32, Dtype::sInt64,
                      Dtypekind::sInt, int32_t, int32_t, __m256i);

DEFINE_PRIMITIVE_TYPE("uint64", 'u', Dtype::sUInt64, Dtype::sFloat64,
                      Dtypekind::sUInt, uint64_t, uint64_t, __m256i);
DEFINE_PRIMITIVE_TYPE("int64", 'i', Dtype::sInt64, Dtype::sFloat64,
                      Dtypekind::sInt, int64_t, int64_t, __m256i);

DEFINE_PRIMITIVE_TYPE("float32", 'f', Dtype::sFloat32, Dtype::sFloat64,
                      Dtypekind::sFloat, float, float, __m256);
DEFINE_PRIMITIVE_TYPE("float64", 'd', Dtype::sFloat64, Dtype::sFloat64,
                      Dtypekind::sFloat, double, double, __m256d);

#undef DEFINE_PRIMITIVE_TYPE

template <typename T>
constexpr Dtype TypeToDtype = PrimitiveType<std::remove_const<T>>::sDtype;

template <typename F, typename... Args>
inline auto dispatch_all_types(Dtype dtype, F&& f, Args&&... args) {
    switch (dtype) {
        case Dtype::sBool:
            return std::forward<F>(f)(PrimitiveType<bool>{},
                                      std::forward<Args>(args)...);
        case Dtype::sUInt8:
            return std::forward<F>(f)(PrimitiveType<uint8_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sInt8:
            return std::forward<F>(f)(PrimitiveType<int8_t>{},
                                      std::forward<Args>(args)...);

        case Dtype::sUInt16:
            return std::forward<F>(f)(PrimitiveType<uint16_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sInt16:
            return std::forward<F>(f)(PrimitiveType<int16_t>{},
                                      std::forward<Args>(args)...);

        case Dtype::sUInt32:
            return std::forward<F>(f)(PrimitiveType<uint32_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sInt32:
            return std::forward<F>(f)(PrimitiveType<int32_t>{},
                                      std::forward<Args>(args)...);

        case Dtype::sUInt64:
            return std::forward<F>(f)(PrimitiveType<uint64_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sInt64:
            return std::forward<F>(f)(PrimitiveType<int64_t>{},
                                      std::forward<Args>(args)...);

        case Dtype::sFloat32:
            return std::forward<F>(f)(PrimitiveType<float>{},
                                      std::forward<Args>(args)...);
        default:
            return std::forward<F>(f)(PrimitiveType<double>{},
                                      std::forward<Args>(args)...);
    }
}
template <typename F, typename... Args>
inline auto dispatch_all_numeric_types(Dtype dtype, F&& f, Args&&... args) {
    switch (dtype) {
        case Dtype::sBool:
            THROW_ERROR_DETAILED(DtypeError, "Boolean is not a numeric type");
        case Dtype::sUInt8:
            return std::forward<F>(f)(PrimitiveType<uint8_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sInt8:
            return std::forward<F>(f)(PrimitiveType<int8_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sUInt16:
            return std::forward<F>(f)(PrimitiveType<uint16_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sInt16:
            return std::forward<F>(f)(PrimitiveType<int16_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sUInt32:
            return std::forward<F>(f)(PrimitiveType<uint32_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sInt32:
            return std::forward<F>(f)(PrimitiveType<int32_t>{},
                                      std::forward<Args>(args)...);

        case Dtype::sUInt64:
            return std::forward<F>(f)(PrimitiveType<uint64_t>{},
                                      std::forward<Args>(args)...);
        case Dtype::sInt64:
            return std::forward<F>(f)(PrimitiveType<int64_t>{},
                                      std::forward<Args>(args)...);

        case Dtype::sFloat32:
            return std::forward<F>(f)(PrimitiveType<float>{},
                                      std::forward<Args>(args)...);
        default:
            return std::forward<F>(f)(PrimitiveType<double>{},
                                      std::forward<Args>(args)...);
    }
}
template <typename F, typename... Args>
inline auto dispatch_fp_types(Dtype dtype, F&& f, Args&&... args) {
    switch (dtype) {
        case Dtype::sFloat32:
            return std::forward<F>(f)(PrimitiveType<float>{},
                                      std::forward<Args>(args)...);
        case Dtype::sFloat64:
            return std::forward<F>(f)(PrimitiveType<double>{},
                                      std::forward<Args>(args)...);
        default:
            return;
    }
}

template <typename F_float, typename F_int, typename... Args>
inline auto dispatch_fp_int_types(Dtype dtype, F_float&& f, F_int&& fi,
                                  Args&&... args) {
    switch (dtype) {
        case Dtype::sBool:
            THROW_ERROR_DETAILED(DtypeError, "Boolean is not a numeric type");
        case Dtype::sUInt8:
            return std::forward<F_int>(fi)(PrimitiveType<uint8_t>{},
                                           std::forward<Args>(args)...);
        case Dtype::sInt8:
            return std::forward<F_int>(fi)(PrimitiveType<int8_t>{},
                                           std::forward<Args>(args)...);

        case Dtype::sUInt16:
            return std::forward<F_int>(fi)(PrimitiveType<uint16_t>{},
                                           std::forward<Args>(args)...);
        case Dtype::sInt16:
            return std::forward<F_int>(fi)(PrimitiveType<int16_t>{},
                                           std::forward<Args>(args)...);

        case Dtype::sUInt32:
            return std::forward<F_int>(fi)(PrimitiveType<uint32_t>{},
                                           std::forward<Args>(args)...);
        case Dtype::sInt32:
            return std::forward<F_int>(fi)(PrimitiveType<int32_t>{},
                                           std::forward<Args>(args)...);

        case Dtype::sUInt64:
            return std::forward<F_int>(fi)(PrimitiveType<uint64_t>{},
                                           std::forward<Args>(args)...);
        case Dtype::sInt64:
            return std::forward<F_int>(fi)(PrimitiveType<int64_t>{},
                                           std::forward<Args>(args)...);

        case Dtype::sFloat32:
            return std::forward<F_float>(f)(PrimitiveType<float>{},
                                            std::forward<Args>(args)...);
        default:
            return std::forward<F_float>(f)(PrimitiveType<double>{},
                                            std::forward<Args>(args)...);
    }
}

template <typename T>
constexpr bool IsFloatingPointV = std::is_floating_point<T>::value;
template <typename T>
constexpr bool IsIntegerV =
    std::is_integral<T>::value && !std::is_same<T, bool>::value;

inline Dtype GetDtypeFromNumpyInt(int npdtype) {
    switch (npdtype) {
        case 0:
            return Dtype::sBool;
        case 1:
            return Dtype::sInt8;
        case 2:
            return Dtype::sUInt8;
        case 3:
            return Dtype::sInt16;
        case 4:
            return Dtype::sUInt16;
        case 5:
            return Dtype::sInt32;
        case 6:
            return Dtype::sUInt32;
        case 7:
            return Dtype::sInt64;
        case 8:
            return Dtype::sUInt64;
        case 11:
            return Dtype::sFloat32;
        case 12:
            return Dtype::sFloat64;
        default:
            break;
    }
    THROW_ERROR_DETAILED(DtypeError, npdtype, " is not a supported dtype");
}
inline int get_np_type_numFromDtype(Dtype dtype) {
    switch (dtype) {
        case Dtype::sBool:
            return 0;
        case Dtype::sInt8:
            return 1;
        case Dtype::sUInt8:
            return 2;
        case Dtype::sInt16:
            return 3;
        case Dtype::sUInt16:
            return 4;
        case Dtype::sInt32:
            return 5;
        case Dtype::sUInt32:
            return 6;
        case Dtype::sInt64:
            return 7;
        case Dtype::sUInt64:
            return 8;
        case Dtype::sFloat32:
            return 11;
        case Dtype::sFloat64:
            return 12;
    }
}

inline long GetDtypeSize(Dtype dtype) {
    struct Pair {
        Dtype dtype;
        const long size;
    };

    static const Pair sMapping[] = {
        {Dtype::sBool, sizeof(bool)},     {Dtype::sUInt8, sizeof(uint8_t)},
        {Dtype::sInt8, sizeof(int8_t)},

        {Dtype::sInt16, sizeof(int16_t)}, {Dtype::sUInt16, sizeof(uint16_t)},

        {Dtype::sInt32, sizeof(int32_t)}, {Dtype::sUInt32, sizeof(uint32_t)},

        {Dtype::sInt64, sizeof(int64_t)}, {Dtype::sUInt64, sizeof(uint64_t)},

        {Dtype::sFloat32, sizeof(float)}, {Dtype::sFloat64, sizeof(double)},

    };

    for (const Pair& pair : sMapping) {
        if (dtype == pair.dtype) {
            return pair.size;
        }
    }
}

typedef struct {
    int alignment;
    int dtype_size;
    int jump;
} alignemnt_information;

inline alignemnt_information getAlignment(Dtype dtype) {
    alignemnt_information info;
    switch (dtype) {
        case Dtype::sBool:
            info = {32, 1, 32};
            return info;
        case Dtype::sInt8:
            info = {32, 1, 32};
            return info;
        case Dtype::sUInt8:
            info = {32, 1, 32};
            return info;
        case Dtype::sUInt16:
            info = {32, 2, 16};
            return info;
        case Dtype::sInt16:
            info = {32, 2, 16};
            return info;
        case Dtype::sUInt32:
            info = {32, 4, 8};
            return info;
        case Dtype::sInt32:
            info = {32, 4, 8};
            return info;
        case Dtype::sUInt64:
            info = {32, 8, 4};
            return info;
        case Dtype::sInt64:
            info = {32, 8, 4};
            return info;
        case Dtype::sFloat32:
            info = {32, 4, 8};
            return info;
        default:
            info = {32, 8, 4};
            return info;
    }
}

inline Dtype promote_dtype(Dtype dt1, Dtype dt2, bool float_only = false) {
    if (dt1 == Dtype::sBool) {
        dt1 = Dtype::sUInt8;
    }
    if (dt2 == Dtype::sBool) {
        dt2 = Dtype::sUInt8;
    }
    Dtype out_dt = dt1;

    if (float_only) {
        if (dt1 == dt2 && dt1 == Dtype::sFloat32) {
            return dt1;
        } else if (dt1 == dt2 && dt1 == Dtype::sFloat64) {
            return dt2;
        } else if (dt1 == Dtype::sFloat32) {
            if (dt2 == Dtype::sInt16 || dt2 == Dtype::sUInt16 ||
                dt2 == Dtype::sInt8 || dt2 == Dtype::sUInt8) {
                return Dtype::sFloat32;
            } else {
                return Dtype::sFloat64;
            }
        } else if (dt2 == Dtype::sFloat32) {
            if (dt1 == Dtype::sInt16 || dt1 == Dtype::sUInt16 ||
                dt1 == Dtype::sInt8 || dt1 == Dtype::sUInt8) {
                return Dtype::sFloat32;
            } else {
                return Dtype::sFloat64;
            }
        } else {
            return Dtype::sFloat64;
        }
    }

    if (dt1 == dt2) {
        return dt1;
    }

    dispatch_all_numeric_types(dt1, [&](auto pt1) {
        dispatch_all_numeric_types(dt2, [&](auto pt2) {
            Dtypekind dt1_k = decltype(pt1)::sKind;
            Dtypekind dt2_k = decltype(pt2)::sKind;
            Dtype dt1_next = decltype(pt1)::nextDtype;
            Dtype dt2_next = decltype(pt2)::nextDtype;

            if (dt1_k == dt2_k) {
                out_dt = std::max(dt1, dt2);
            } else if (dt1_k == Dtypekind::sUInt && dt2_k == Dtypekind::sInt) {
                out_dt = std::max(dt1_next, dt2);
            } else if (dt1_k == Dtypekind::sInt && dt2_k == Dtypekind::sUInt) {
                out_dt = std::max(dt2_next, dt1);
            } else if (dt1_k == Dtypekind::sUInt &&
                       dt2_k == Dtypekind::sFloat) {
                if (dt1 == Dtype::sUInt8 || dt1 == Dtype::sUInt16) {
                    dt1 = Dtype::sFloat32;
                } else {
                    dt1 = Dtype::sFloat64;
                }
                out_dt = std::max(dt1, dt2);
            } else if (dt1_k == Dtypekind::sFloat &&
                       dt2_k == Dtypekind::sUInt) {
                if (dt2 == Dtype::sUInt8 || dt2 == Dtype::sUInt16) {
                    dt2 = Dtype::sFloat32;
                } else {
                    dt2 = Dtype::sFloat64;
                }
                out_dt = std::max(dt1, dt2);
            } else if (dt1_k == Dtypekind::sInt && dt2_k == Dtypekind::sFloat) {
                if (dt1 == Dtype::sInt8 || dt1 == Dtype::sInt16) {
                    dt1 = Dtype::sFloat32;
                } else {
                    dt1 = Dtype::sFloat64;
                }
                out_dt = std::max(dt1, dt2);
            } else if (dt1_k == Dtypekind::sFloat && dt2_k == Dtypekind::sInt) {
                if (dt2 == Dtype::sInt8 || dt2 == Dtype::sInt16) {
                    dt2 = Dtype::sFloat32;
                } else {
                    dt2 = Dtype::sFloat64;
                }
                out_dt = std::max(dt1, dt2);
            } else {
                out_dt = std::max(dt1, dt2);
            }
        });
    });

    return out_dt;
}

inline Dtype min_type(double d) {
    if ((double)((float)d) == d) {
        return Dtype::sFloat32;
    }
    return Dtype::sFloat64;
}

inline Dtype min_type(long d) {
    if (d > 0) {
        if (d < 250) {
            return Dtype::sUInt8;
        } else if (d < 65535) {
            return Dtype::sUInt16;
        } else if (d < 4294967295) {
            return Dtype::sUInt32;
        } else {  // 18446744073709551615
            return Dtype::sUInt64;
        }
    } else {
        if (-128 < d && d < 127) {
            return Dtype::sInt8;
        } else if (-32768 < d && d < 32767) {
            return Dtype::sInt16;
        } else if (-2147483648 < d && d < 2147483647) {
            return Dtype::sInt32;
        } else {
            return Dtype::sInt64;
        }
    }

    return Dtype::sInt64;
}
