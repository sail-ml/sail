#pragma once
#include "Tensor.h"
#include "dtypes.h"
#include "modules/module.h"
namespace sail {
namespace optimizers {

using TensorVector = std::vector<Tensor>;

class Optimizer {
   public:
    TensorVector params;
    long steps = 0;

    explicit Optimizer() = default;
    virtual void update(){};

    void track_module(modules::Module& mod) {
        for (const auto& t : mod.params) {
            params.push_back(t);
        }
    }

    virtual void step(){};
};
}  // namespace optimizers
}  // namespace sail
