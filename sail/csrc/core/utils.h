#pragma once

#include <malloc.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include <boost/range/counting_range.hpp>
#include <boost/range/irange.hpp>
#include "exception.h"
#include "types.h"

namespace sail {

inline long roundUp(long numToRound, long multiple) {
    if (multiple == 0) return numToRound;

    long remainder = numToRound % multiple;
    if (remainder == 0) return numToRound;

    return numToRound + multiple - remainder;
}

inline void* _malloc_align(long numel, long alignment, long dtype_size) {
    long size = dtype_size * numel;
    if (size % alignment != 0) {
        size = roundUp(size, alignment);
    }
    void* pv;
#if defined(_ISOC11_SOURCE)
    pv = aligned_alloc(alignment, size);
#else
    pv = memalign(alignment, size);
#endif

    if (pv == nullptr) {
        throw SailCError("Allocation failed");
    }

    return pv;
}

inline void* _realloc_align(void* src, long numel, long alignment,
                            long dtype_size) {
    void* aligned = _malloc_align(numel, alignment, dtype_size);
    if (aligned == nullptr) {
        throw SailCError("Allocation failed");
    }
    memcpy(aligned, src, dtype_size * numel);
    return aligned;
}
inline void* _calloc_align(long numel, long alignment, long dtype_size) {
    void* aligned = _malloc_align(numel, alignment, dtype_size);
    if (aligned == nullptr) {
        std::cout << "ALLOC FAIL" << std::endl;
    }
    memset(aligned, 0, dtype_size * numel);
    return aligned;
}

inline int prod_size_vector(const TensorSize size) {
    int s = 1;
    for (int v : size) {
        s *= v;
    }
    return s;
}

inline std::string getVectorString(const TensorSize vector) {
    if (vector.size() != 0) {
        std::stringstream result;
        std::copy(vector.begin(), vector.end(),
                  std::ostream_iterator<int>(result, ", "));
        std::string x = result.str();
        x.pop_back();
        x.pop_back();
        // std::string  shape_string("(");
        return std::string("(") + x + std::string(")");
    }
    return std::string("()");
}
inline std::string getVectorString(const std::vector<std::string> vector) {
    if (vector.size() != 0) {
        std::stringstream result;
        std::copy(vector.begin(), vector.end(),
                  std::ostream_iterator<std::string>(result, ", "));
        std::string x = result.str();
        x.pop_back();
        x.pop_back();
        // std::string  shape_string("(");
        return std::string("(") + x + std::string(")");
    }
    return std::string("()");
}

template <typename Integer>
boost::integer_range<Integer> irange(Integer first, Integer last) {
    SAIL_CHECK(first <= last);
    return boost::integer_range<Integer>(first, last);
}

}  // namespace sail
