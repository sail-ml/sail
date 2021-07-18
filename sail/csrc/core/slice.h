#pragma once
#include <cstdarg>
#include <iostream>
#include <vector>
#include "utils.h"

namespace sail {
class Slice {
   public:
    std::vector<std::vector<long>> slices;
    Slice(std::vector<std::vector<long>> slices) : slices(std::move(slices)){};
    Slice(std::vector<long> slices) : slices({slices}){};
    Slice(std::vector<long> slice, int axis) {
        for (int i = 0; i < axis; i++) {
            slices.push_back({});  // NOLINT
        }
        slices.emplace_back(slice);
    };

    void print() {
        for (const auto& slice : slices) {
            std::cout << getVectorString(slice) << " ";
        }
        std::cout << std::endl;
    }

    long size() { return slices.size(); }
};
}  // namespace sail