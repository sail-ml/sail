#include <cmath>
#include <cstdio>
#include <vector>
#include "Tensor.h"
#include "exception.h"
#include "factories.h"
#include "kaiming.h"
#include "utils.h"

namespace sail {

namespace initializers {

Tensor xavier_uniform(Tensor input, double gain) {
    std::tuple<long, long> fans = calculate_fan_in_out(input);
    long fan1 = std::get<0>(fans);
    long fan2 = std::get<1>(fans);

    double bound = gain * std::sqrt(6.0 / (fan1 + fan2));

    input = random::uniform_fill(input, -bound, bound);
    return input;
}

Tensor xavier_normal(Tensor input, double gain) {
    std::tuple<long, long> fans = calculate_fan_in_out(input);
    long fan1 = std::get<0>(fans);
    long fan2 = std::get<1>(fans);

    double std = gain * std::sqrt(2.0 / (fan1 + fan2));

    input = random::normal_fill(input, 0, std * std);
    return input;
}

}  // namespace initializers

}  // namespace sail