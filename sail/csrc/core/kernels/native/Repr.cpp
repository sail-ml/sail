// allow-no-header allow-class-definitions

#include "kernels/Repr.h"
#include "Tensor.h"
#include "dtypes.h"
#include "kernels/dispatch.h"
#include "kernels/native/loops.h"
#include "tensor_shape.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "Tensor.h"
#include "dtypes.h"
#include "factories.h"
#include "tensor_shape.h"

namespace sail {

namespace internal {

namespace {

inline int GetNDigits(int64_t value) {
    int digits = 0;
    while (value != 0) {
        value /= 10;
        ++digits;
    }
    return digits;
}

inline void PrintNTimes(std::ostream& os, char c, int n) {
    while (n-- > 0) {
        os << c;
    }
}

class FloatFormatter {
   public:
    void Scan(double value) {
        int b_digits = 0;
        if (value < 0) {
            has_minus_ = true;
            ++b_digits;
            value = -value;
        }
        if (value >= 100'000'000) {
            int e_digits = GetNDigits(static_cast<int64_t>(std::log10(value)));
            if (digits_after_e_ < e_digits) {
                digits_after_e_ = e_digits;
            }
        }
        if (digits_after_e_ > 0) {
            return;
        }

        const auto int_frac_parts = IntFracPartsToPrint(value);

        b_digits += GetNDigits(int_frac_parts.first);
        if (digits_before_point_ < b_digits) {
            digits_before_point_ = b_digits;
        }

        const int a_digits = GetNDigits(int_frac_parts.second) - 1;
        if (digits_after_point_ < a_digits) {
            digits_after_point_ = a_digits;
        }
    }

    void Print(std::ostream& os, double value) {
        if (digits_after_e_ > 0) {
            int width = 12 + (has_minus_ ? 1 : 0) + digits_after_e_;
            if (has_minus_ && !std::signbit(value)) {
                os << ' ';
                --width;
            }
            os << std::scientific << std::left << std::setw(width)
               << std::setprecision(8) << value;
        } else {
            const auto int_frac_parts = IntFracPartsToPrint(value);
            const int a_digits = GetNDigits(int_frac_parts.second) - 1;
            os << std::fixed << std::right
               << std::setw(digits_before_point_ + a_digits + 1)
               << std::setprecision(a_digits) << std::showpoint << value;
            PrintNTimes(os, ' ', digits_after_point_ - a_digits);
        }
    }

   private:
    // code comment
    static std::pair<int64_t, int64_t> IntFracPartsToPrint(double value) {
        double int_part;
        const double frac_part = std::modf(value, &int_part);

        auto shifted_frac_part =
            static_cast<int64_t>((std::abs(frac_part) + 1) * 100'000'000);
        while ((shifted_frac_part % 10) == 0) {
            shifted_frac_part /= 10;
        }

        return {static_cast<int64_t>(int_part), shifted_frac_part};
    }

    int digits_before_point_ = 1;
    int digits_after_point_ = 0;
    int digits_after_e_ = 0;
    bool has_minus_ = false;
};

class IntFormatter {
   public:
    void Scan(int64_t value) {
        int digits = 0;
        if (value < 0) {
            ++digits;
            value = -value;
        }
        digits += GetNDigits(value);
        if (max_digits_ < digits) {
            max_digits_ = digits;
        }
    }

    void Print(std::ostream& os, int64_t value) const {
        os << std::setw(max_digits_) << std::right << value;
    }

   private:
    int max_digits_ = 1;
};

class BoolFormatter {
   public:
    void Scan(int64_t value) { max_digits_ = 5; }

    void Print(std::ostream& os, int64_t value) const {
        std::string print_val = "True";
        if (value == 0) {
            print_val = "False";
        }
        os << std::setw(max_digits_) << std::right << print_val;
    }

   private:
    int max_digits_ = 1;
};

template <typename T>
using Formatter = std::conditional_t<
    IsFloatingPointV<T>, FloatFormatter,
    std::conditional_t<IsIntegerV<T>, IntFormatter, BoolFormatter>>;

class ReprKernel {
   public:
    void execute(const Tensor& t1, std::ostream& os) {
        dispatch_all_types(t1.get_dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Formatter<T> formatter;

            T* data = (T*)t1.get_data();

            // scan all elements
            TensorShape shape = t1.get_shape();
            shape.recompute();
            long numel = shape.numel();
            if (t1.is_view()) {
                for (int i = 0; i < numel; i++) {
                    T value = data[shape.d_ptr];
                    formatter.Scan(value);
                    shape.next();
                }
                shape.reset();
            } else {
                for (int i = 0; i < numel; i++) {
                    T value = data[i];
                    formatter.Scan(value);
                }
            }
            os << "tensor(";
            if (t1.is_scalar()) {
                T x = *(T*)t1.get_data();
                formatter.Scan(x);
                formatter.Print(os, x);
            } else {
                bool should_abbreviate = t1.get_shape().numel() > kThreshold;
                ArrayReprRecursive<T>(t1, formatter, 8, os, should_abbreviate);
            }
            if (!t1.is_scalar()) {
                os << ", shape=" << t1.get_shape().get_string();
            }
            os << ")";
        });
    }

   private:
    static constexpr int kMaxItemNumPerLine = 10;
    static constexpr int64_t kThreshold = 1000;
    static constexpr int64_t kEdgeItems = 3;

    template <typename T>
    void ArrayReprRecursive(const Tensor& tensor, Formatter<T>& formatter,
                            size_t indent, std::ostream& os,
                            bool abbreviate = false) const {
        long ndim = tensor.get_shape().ndim();
        if (tensor.is_scalar()) {
            formatter.Print(os, *(T*)tensor.get_data());
            return;
        }
        auto print_indent = [ndim, indent, &os](int64_t i) {
            if (i != 0) {
                os << "";
                if (ndim > 1 || i % kMaxItemNumPerLine == 0) {
                    if (ndim == 1) {
                        PrintNTimes(os, '\n', ndim);
                    } else {
                        PrintNTimes(os, '\n', ndim - 1);
                    }
                    PrintNTimes(os, ' ', indent);
                } else {
                    os << ' ';
                }
            }
        };
        os << "[";

        long size = tensor.get_shape().shape[0];
        T* data = (T*)tensor.get_data();

        TensorShape shape = tensor.get_shape();

        if (abbreviate && size > kEdgeItems * 2) {
            for (int64_t i = 0; i < kEdgeItems; ++i) {
                print_indent(i);
                Tensor t = tensor[i];
                ArrayReprRecursive<T>(t, formatter, indent + 1, os, abbreviate);
            }
            print_indent(1);
            os << "...";
            print_indent(1);
            for (int64_t i = size - 3; i < size; ++i) {
                print_indent(i);
                Tensor t = tensor[i];
                ArrayReprRecursive<T>(t, formatter, indent + 1, os, abbreviate);
            }
        } else {
            for (long i = 0; i < size; ++i) {
                print_indent(i);
                Tensor t = tensor[i];
                ArrayReprRecursive<T>(t, formatter, indent + 1, os, abbreviate);
            }
        }
        os << "]";
    }
};

void repr_kernel(const Tensor& t1, std::ostream& os) {
    ReprKernel().execute(t1, os);
}

}  // namespace
REGISTER_ONLY_NATIVE_DISPATCH(repr_stub, &repr_kernel);

}  // namespace internal

}  // namespace sail