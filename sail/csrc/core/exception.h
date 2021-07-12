#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

inline void MakeMessageImpl(std::ostringstream& /*os*/) {}

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, int8_t first, const Args&... args);

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, uint8_t first,
                     const Args&... args);

template <typename Arg, typename... Args>
void MakeMessageImpl(std::ostringstream& os, const Arg& first,
                     const Args&... args) {
    os << first;
    MakeMessageImpl(os, args...);
}

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, int8_t first,
                     const Args&... args) {
    os << static_cast<int>(first);
    MakeMessageImpl(os, args...);
}

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, uint8_t first,
                     const Args&... args) {
    os << static_cast<unsigned int>(first);
    MakeMessageImpl(os, args...);
}

template <typename... Args>
std::string MakeMessage(const Args&... args) {
    std::ostringstream os;
    os << std::boolalpha;
    MakeMessageImpl(os, args...);
    return os.str();
}

class SailCError : public std::runtime_error {
   public:
    template <typename... Args>
    explicit SailCError(const Args&... args)
        : runtime_error{MakeMessage(args...)} {}
};

// Error on using invalid contexts.
class DimensionError : public SailCError {
   public:
    using SailCError::SailCError;
};
// Error on using invalid contexts.
class DtypeError : public SailCError {
   public:
    using SailCError::SailCError;
};
// Error on using invalid contexts.
class TypeError : public SailCError {
   public:
    using SailCError::SailCError;
};

// #define SAIL_ASSERT(cond, ...)
//   if (C10_UNLIKELY_OR_CONST(!(cond))) {
//     ::c10::detail::torchCheckFail(
//         __func__,
//         __FILE__,
//         static_cast<uint32_t>(__LINE__),
//         #cond "INTERNAL ASSERT FAILED at" C10_STRINGIZE(__FILE__));
//   }
#define SAIL_CHECK(cond, ...)          \
    if (!(cond)) {                     \
        throw SailCError(__VA_ARGS__); \
    }
#define SAIL_CHECK_LINE(cond, ...)                                      \
    if (!(cond)) {                                                      \
        throw SailCError(__FILE__, ":", __LINE__, "\n", ##__VA_ARGS__); \
    }
#define SAIL_TYPE_CHECK_2(a, b)                                                \
    if (a.get_dtype() != b.get_dtype()) {                                      \
        throw TypeError("Tensor types do not match. Recieved ", a.get_dtype(), \
                        " and ", b.get_dtype());                               \
    }

#define THROW_ERROR(err_t, ...) throw err_t(__VA_ARGS__)
#define THROW_ERROR_DETAILED(err_t, ...)               \
    throw err_t(__VA_ARGS__, "\n", "File: ", __FILE__, \
                "\nLine: ", __LINE__)  // throw err_t("Error occured at ",
                                       // __FILE__, ":", __LINE__, ":\n",
                                       //             ##__VA_ARGS__)