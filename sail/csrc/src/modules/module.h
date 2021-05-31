#pragma once
#include "../dtypes.h"
namespace sail {
namespace modules {

inline Dtype default_dtype = Dtype::sFloat32;

class Module {
   public:
    virtual explicit Module(){};
    virtual void forward(){};
};
}  // namespace modules
}  // namespace sail
