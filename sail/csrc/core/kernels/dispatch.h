// allow-no-source allow-impl-in-header

#pragma once

#include <atomic>

namespace sail {
namespace internal {

enum class CPUCapability { DEFAULT = 0, AVX = 1, NUM_OPTIONS };
enum class KernelCapability { DEFAULT = 0, AVX = 1, NUM_OPTIONS };

template <typename FnPtr>
struct fcn_table {
    FnPtr DEFAULT = nullptr;
    FnPtr AVX = nullptr;
};

inline CPUCapability get_cpu_capability() {
#ifdef USE_AVX
    return CPUCapability::AVX;
#endif
    return CPUCapability::DEFAULT;
}

template <typename FnPtr, typename T>
struct DispatchStub;

template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
    using FnPtr = rT (*)(Args...);

    DispatchStub() = default;
    DispatchStub(const DispatchStub&) = delete;
    DispatchStub& operator=(const DispatchStub&) = delete;

   public:
    template <typename... ArgTypes>
    inline rT operator()(ArgTypes&&... args) {
        if (USE == nullptr) {  // optimize after first call
            FnPtr call = DEFAULT;
#ifdef USE_AVX
            if (AVX != nullptr) {
                call = AVX;
            }
#endif
            USE = call;
        }
        return (*USE)(args...);
    }

    FnPtr USE = nullptr;
    static FnPtr DEFAULT;
#ifdef USE_AVX
    static FnPtr AVX;
#endif
};

#define DECLARE_DISPATCH(fn, name)             \
    struct name : DispatchStub<fn, name> {     \
        name() = default;                      \
        name(const name&) = delete;            \
        name& operator=(const name&) = delete; \
    };                                         \
    extern struct name name

#define DEFINE_DISPATCH(name) struct name name

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
    template <>                                \
    decltype(fn) DispatchStub<decltype(fn), struct name>::arch = fn;

#ifdef USE_AVX

#define REGISTER_AVX_DISPATCH(name, fn) \
    template <>                         \
    decltype(fn) DispatchStub<decltype(fn), struct name>::AVX = fn;

#else

#define REGISTER_AVX_DISPATCH(name, fn) ;
#endif

#define REGISTER_ONLY_NATIVE_DISPATCH(name, fn) \
    REGISTER_ARCH_DISPATCH(name, DEFAULT, fn);  \
    REGISTER_AVX_DISPATCH(name, static_cast<decltype(fn)>(nullptr));
}  // namespace internal
}  // namespace sail