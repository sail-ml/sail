#pragma once

#include <stdio.h>
#include <cmath>
#include <vector>
#include "Tensor.h"
#include "exception.h"
#include "factories.h"

namespace sail {

namespace initializers {

inline double get_gain(std::string nonlin) {
    if (nonlin == "linear") {
        return 1.0;
    }
    if (nonlin == "conv2d") {
        return 1.0;
    }
    if (nonlin == "sigmoid") {
        return 1.0;
    }
    if (nonlin == "tanh") {
        return 5.0 / 3;
    }
    if (nonlin == "relu") {
        return std::sqrt(2.0);
    }
    if (nonlin == "leaky_relu") {
        return std::sqrt(2.0 / (1 + 0.01 * 0.01));
    }

    THROW_ERROR_DETAILED(SailCError, "Nonlinearity not supported");
}

inline std::tuple<long, long> calculate_fan_in_out(Tensor& input) {
    if (input.get_ndim() < 2) {
        THROW_ERROR(SailCError,
                    "fan in and out cannot be calculated for a tensor with "
                    "less than 2 dimensions");
    }
    long num_input_fmaps = input.get_shape()[1];
    long num_output_fmaps = input.get_shape()[0];
    long receptive_field_size = 1;
    if (input.get_ndim() > 2) {
        for (int i = 2; i < input.get_ndim(); i++) {
            receptive_field_size *= input.get_shape()[i];
        }
    }
    long fan_in = num_input_fmaps * receptive_field_size;
    long fan_out = num_output_fmaps * receptive_field_size;
    return std::make_tuple(fan_in, fan_out);
}

}  // namespace initializers

}  // namespace sail
