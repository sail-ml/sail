#pragma once
namespace sail {
namespace modules {

class Module {
   public:
    virtual explicit Module(){};
    virtual void forward(){};
};
}  // namespace modules
}  // namespace sail
