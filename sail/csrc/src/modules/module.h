#pragma once
namespace sail {
namespace modules {
class Module {
   public:
    explicit Module();
    virtual void forward();
};
}  // namespace modules
}  // namespace sail
