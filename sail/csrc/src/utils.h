#pragma once

#include <iostream>
#include <iterator>
#include <vector>
#include "types.h"

inline bool isAlignedAs(const void* p, const int8_t alignment) {
    return ((int8_t)(p) & ((alignment)-1)) == 0;
}

inline void* _malloc_align(int numel, int alignment, int dtype_size) {
    return aligned_alloc(alignment, dtype_size * numel);
}

inline void* _realloc_align(void* src, int numel, int alignment,
                            int dtype_size) {
    // void* aligned = _mm_malloc(dtype_size * numel, alignment);
    void* aligned = _malloc_align(numel, alignment, dtype_size);
    // void* aligned = static_cast<void*>(new char[dtype_size * numel,
    // alignment]);
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