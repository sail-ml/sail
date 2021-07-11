#pragma once

#include <malloc.h>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <vector>

#include "exception.h"
#include "types.h"

inline bool isAlignedAs(const void* p, const int8_t alignment) {
    return ((int8_t)(p) & ((alignment)-1)) == 0;
}

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

    if (pv == NULL) {
        throw SailCError("Allocation failed");
    }

    return pv;
}

inline void* _realloc_align(void* src, long numel, long alignment,
                            long dtype_size) {
    void* aligned = _malloc_align(numel, alignment, dtype_size);
    if (aligned == NULL) {
        throw SailCError("Allocation failed");
    }
    memcpy(aligned, src, dtype_size * numel);
    return aligned;
}
inline void* _calloc_align(long numel, long alignment, long dtype_size) {
    void* aligned = _malloc_align(numel, alignment, dtype_size);
    if (aligned == NULL) {
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

// inline bool