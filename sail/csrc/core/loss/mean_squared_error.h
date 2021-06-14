#include "Tensor.h"
#include "modules/module.h"

namespace sail {
namespace loss {

class MeanSquaredError : public sail::modules::Module {
   public:
    explicit MeanSquaredError(){};

    Tensor forward(Tensor& logits, Tensor& targets);
};

}  // namespace loss
}  // namespace sail