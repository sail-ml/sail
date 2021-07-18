#include "kaiming.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include "Tensor.h"
#include "exception.h"
#include "factories.h"
#include "utils.h"

namespace sail {

namespace initializers {

Tensor kaiming_uniform(Tensor input, std::string mode, std::string nonlin) {
    std::tuple<long, long> fans = calculate_fan_in_out(input);
    long fan;
    if (mode == "fan_in") {
        fan = std::get<0>(fans);
    } else if (mode == "fan_out") {
        fan = std::get<1>(fans);
    } else {
        THROW_ERROR(SailCError, mode,
                    " is not a supported kaiming uniform mode");
    }

    double gain = get_gain(nonlin);
    double std = gain / std::sqrt((double)fan);
    double bound = std::sqrt(3.0) * std;

    input = random::uniform_fill(input, -bound, bound);
    return input;
}

Tensor kaiming_normal(Tensor input, std::string mode, std::string nonlin) {
    std::tuple<long, long> fans = calculate_fan_in_out(input);
    long fan;
    if (mode == "fan_in") {
        fan = std::get<0>(fans);
    } else if (mode == "fan_out") {
        fan = std::get<1>(fans);
    } else {
        THROW_ERROR(SailCError, mode,
                    " is not a supported kaiming uniform mode");
    }

    double gain = get_gain(nonlin);
    double std = gain / std::sqrt(fan);

    input = random::normal_fill(input, 0, std * std);
    return input;
}

}  // namespace initializers

}  // namespace sail