#pragma once

#include <immintrin.h>
#include <cstring>
// #include "Tensor.h"
#include "error.h"
#include "utils.h"

// using Tensor = sail::Tensor;

enum class Dtype {
    sBool = 1,
    sInt8,
    sInt16,
    sInt32,
    sInt64,
    sUInt8,
    // sFloat16,
    sFloat32,
    sFloat64,
};

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

#define DEFINE_PRIMITIVE_TYPE(name, code, dtype, kind, t, dst, avx) \
    template <>                                                     \
    struct PrimitiveType<t> {                                       \
        using type = t;                                             \
        using device_storage_type = dst;                            \
        using avx_type = avx;                                       \
        static constexpr char sCharCode = code;                     \
        static constexpr Dtype sDtype = dtype;                      \
        static constexpr int64_t sElementSize = sizeof(type);       \
        static constexpr Dtypekind sKind = kind;                    \
        static const char* GetName() { return name; }               \
    }

// TODO(niboshi): Char codes are mapped according to current development
// environment. They should be remapped depending on the executing environment,
// // as in NumPy.
// DEFINE_PRIMITIVE_TYPE("bool", '?', Dtype::sBool, Dtypekind::sBool, bool,
// bool,
//                       __m256i, int8_avx{});
// DEFINE_PRIMITIVE_TYPE("int8", 'b', Dtype::sInt8, Dtypekind::sInt, int8_t,
//                       int8_t, __m256i, int8_avx{});
// DEFINE_PRIMITIVE_TYPE("int16", 'h', Dtype::sInt16, Dtypekind::sInt, int16_t,
//                       int16_t, __m256i, int16_avx{});
DEFINE_PRIMITIVE_TYPE("int32", 'i', Dtype::sInt32, Dtypekind::sInt, int32_t,
                      int32_t, __m256i);
// DEFINE_PRIMITIVE_TYPE("int64", 'l', Dtype::sInt64, Dtypekind::sInt, int64_t,
//                       int64_t, __m256i, int64_avx{});
// DEFINE_PRIMITIVE_TYPE("uint8", 'B', Dtype::sUInt8, Dtypekind::sUInt, uint8_t,
//                       uint8_t, __m256i, int_avx{});
// CHAINERX_DEFINE_PRIMITIVE_TYPE("float16", 'e', Dtype::sFloat16,
// Dtypesind::sFloat, chainerx::Float16, uint16_t);
DEFINE_PRIMITIVE_TYPE("float32", 'f', Dtype::sFloat32, Dtypekind::sFloat, float,
                      float, __m256);
DEFINE_PRIMITIVE_TYPE("float64", 'd', Dtype::sFloat64, Dtypekind::sFloat,
                      double, double, __m256d);

#undef DEFINE_PRIMITIVE_TYPE

template <typename T>
constexpr Dtype TypeToDtype = PrimitiveType<std::remove_const<T>>::sDtype;

template <typename F, typename... Args>
inline auto launch_arithmetic(Dtype dtype, F&& f, Args&&... args) {
    switch (dtype) {
        // case Dtype::sInt8:
        //     return std::forward<F>(f)(PrimitiveType<int8_t>{},
        //     std::forward<Args>(args)...);
        // case Dtype::sInt16:
        //     return std::forward<F>(f)(PrimitiveType<int16_t>{},
        //     std::forward<Args>(args)...);
        case Dtype::sInt32:
            return std::forward<F>(f)(PrimitiveType<int32_t>{},
                                      std::forward<Args>(args)...);
        // case Dtype::sInt64:
        //     return std::forward<F>(f)(PrimitiveType<int64_t>{},
        //     std::forward<Args>(args)...);
        // case Dtype::sUInt8:
        //     return std::forward<F>(f)(PrimitiveType<uint8_t>{},
        //     std::forward<Args>(args)...);
        // // case Dtype::sFloat16:
        // //     return std::forward<F>(f)(PrimitiveType<chainerx::Float16>{},
        // std::forward<Args>(args)...);
        case Dtype::sFloat32:
            return std::forward<F>(f)(PrimitiveType<float>{},
                                      std::forward<Args>(args)...);
        case Dtype::sFloat64:
            return std::forward<F>(f)(PrimitiveType<double>{},
                                      std::forward<Args>(args)...);
        default:
            throw DtypeError{"Dtype error in launch arithmetic"};
    }
}

inline char GetCharCode(Dtype dtype) {
    return launch_arithmetic(dtype,
                             [](auto pt) { return decltype(pt)::sCharCode; });
}

inline Dtype GetDtype(const std::string& name) {
    struct Pair {
        const char* name;
        Dtype dtype;
    };

    static const Pair sMapping[] = {// full name
                                    {"bool", Dtype::sBool},
                                    {"int8", Dtype::sInt8},
                                    {"int16", Dtype::sInt16},
                                    {"int32", Dtype::sInt32},
                                    {"int64", Dtype::sInt64},
                                    {"uint8", Dtype::sUInt8},
                                    // {"float16", Dtype::sFloat16},
                                    {"float32", Dtype::sFloat32},
                                    {"float64", Dtype::sFloat64},
                                    // character code
                                    {"?", Dtype::sBool},
                                    {"b", Dtype::sInt8},
                                    {"h", Dtype::sInt16},
                                    {"i", Dtype::sInt32},
                                    {"l", Dtype::sInt64},
                                    {"B", Dtype::sUInt8},
                                    // {"e", Dtype::sFloat16},
                                    {"f", Dtype::sFloat32},
                                    {"d", Dtype::sFloat64}};

    const char* cname = name.c_str();
    for (const Pair& pair : sMapping) {
        if (0 == std::strcmp(pair.name, cname)) {
            return pair.dtype;
        }
    }
    throw DtypeError{"Dtype not found get dtype"};
}

inline Dtype GetDtypeFromNumpyInt(int npdtype) {
    switch (npdtype) {
        case 5:
            return Dtype::sInt32;
        case 11:
            return Dtype::sFloat32;
        case 12:
            return Dtype::sFloat64;
        default:
            break;
    }
    // throw DtypeError{"unsupported NumPy dtype"};
    throw DtypeError{"Dtype not found np int"};
}
inline int get_np_type_numFromDtype(Dtype dtype) {
    switch (dtype) {
        case Dtype::sInt32:
            return 5;
        case Dtype::sFloat32:
            return 11;
        case Dtype::sFloat64:
            return 12;
        default:
            break;
    }
    // std::cout << dtype << std::endl;
    // throw DtypeError{"unsupported NumPy dtype"};
    throw DtypeError{"Dtype not found NP DTYE"};
}

inline long GetDtypeSize(Dtype dtype) {
    struct Pair {
        Dtype dtype;
        const long size;
    };

    static const Pair sMapping[] = {
        // full name
        {Dtype::sBool, sizeof(bool)},     {Dtype::sInt8, sizeof(int8_t)},
        {Dtype::sInt16, sizeof(int16_t)}, {Dtype::sInt32, sizeof(int32_t)},
        {Dtype::sInt64, sizeof(int64_t)}, {Dtype::sUInt8, sizeof(uint8_t)},
        {Dtype::sFloat32, sizeof(float)}, {Dtype::sFloat64, sizeof(double)},

    };

    for (const Pair& pair : sMapping) {
        if (dtype == pair.dtype) {
            return pair.size;
        }
    }
    throw DtypeError{"Dtype not found get size"};
}

typedef struct {
    int alignment;
    int dtype_size;
    int jump;
} alignemnt_information;

inline alignemnt_information getAlignment(Dtype dtype) {
    alignemnt_information info;
    switch (dtype) {
        // case Dtype::sInt8:
        //     return std::forward<F>(f)(PrimitiveType<int8_t>{},
        //     std::forward<Args>(args)...);
        // case Dtype::sInt16:
        //     return std::forward<F>(f)(PrimitiveType<int16_t>{},
        //     std::forward<Args>(args)...);
        // case Dtype::sInt32:
        //     return std::forward<F>(f)(PrimitiveType<int32_t>{},
        //     std::forward<Args>(args)...);
        // case Dtype::sInt64:
        //     return std::forward<F>(f)(PrimitiveType<int64_t>{},
        //     std::forward<Args>(args)...);
        // case Dtype::sUInt8:
        //     return std::forward<F>(f)(PrimitiveType<uint8_t>{},
        //     std::forward<Args>(args)...);
        // // case Dtype::sFloat16:
        // //     return std::forward<F>(f)(PrimitiveType<chainerx::Float16>{},
        // std::forward<Args>(args)...); case Dtype::sFloat32:
        //     return std::forward<F>(f)(PrimitiveType<float>{},
        //     std::forward<Args>(args)...);
        case Dtype::sInt32:
            info = {32, 4, 8};
            return info;
        case Dtype::sFloat32:
            info = {32, 4, 8};
            return info;
        case Dtype::sFloat64:
            info = {32, 8, 4};
            return info;
        default:
            throw DtypeError{"Dtype error GET ALIGNMENT"};
    }
}

// inline bool can_cast(Dtype from, Dtype to){};
// #include "Tensor.h"
