#pragma once

#include <malloc.h>
#include <iostream>
#include <iterator>
#include <vector>

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
    // return malloc(dtype_size * numel);
    // std::cout << dtype_size * numel << ", " << alignment << std::endl;
    // return _mm_malloc(dtype_size * numel, alignment);
    // long v = dtype_size * numel;
    // if ((v & (v - 1)) == 0) {
    //     std::cout << "not two" << std::endl;
    long size = dtype_size * numel;
    if (size % alignment != 0) {
        std::cout << "rounding " << size;
        size = roundUp(size, alignment);
        std::cout << " to " << size << std::endl;
    }
    // }
    // return memalign(alignment, v);
    return aligned_alloc(alignment, size);
}

inline void* _realloc_align(void* src, long numel, long alignment,
                            long dtype_size) {
    void* aligned = _malloc_align(numel, alignment, dtype_size);
    if (aligned == NULL) {
        std::cout << "ALLOC FAIL" << std::endl;
    }
    memcpy(aligned, src, dtype_size * numel);
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
    std::stringstream result;
    std::copy(vector.begin(), vector.end(),
              std::ostream_iterator<int>(result, ", "));
    std::string x = result.str();
    x.pop_back();
    x.pop_back();
    // std::string  shape_string("(");
    return std::string("(") + x + std::string(")");
}

// inline bool