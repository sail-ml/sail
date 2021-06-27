#pragma once

#include <atomic>
// Implements instruction set specific function dispatch.
//
// Kernels that may make use of specialized instruction sets (e.g. AVX) are
// compiled multiple times with different compiler flags (e.g. -mavx). A
// DispatchStub contains a table of function pointers for a kernel. At runtime,
// the fastest available kernel is chosen based on the features reported by
// cpuinfo.
//
// Example:
//
// In native/MyKernel.h:
//   using fn_type = void(*)(const Tensor& x);
//   DECLARE_DISPATCH(fn_type, stub);
//
// In native/MyKernel.cpp
//   DEFINE_DISPATCH(stub);
//
// In native/cpu/MyKernel.cpp:
//   namespace {
//     // use anonymous namespace so that different cpu versions won't conflict
//     void kernel(const Tensor& x) { ... }
//   }
//   REGISTER_DISPATCH(stub, &kernel);
//
// To call:
//   stub(kCPU, tensor);
//
// TODO: CPU instruction set selection should be folded into whatever
// the main dispatch mechanism is.

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
        // FnPtr call_ptr = DEFAULT;
        return (*USE)(args...);
        // return (*USE)(std::forward<ArgTypes>(args)...);
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