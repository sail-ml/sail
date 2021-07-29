// allow-no-source

#pragma once
namespace sail {
namespace modules {
class Layer {
   public:
    explicit Layer();
    virtual void forward();
};
}  // namespace modules
}  // namespace sail
