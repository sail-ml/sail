#pragma once
#include "tensor_shape.h"

namespace sail {

class TensorIterator {
   public:
    std::vector<long> shape;
    std::vector<long> strides;
    std::vector<long> shape_m1;
    std::vector<long> strides_back;
    std::vector<long> coordinates;

    explicit TensorIterator(){};
    TensorIterator(TensorShape& t_shape);

    long numel() const;
    long ndim() const;
    long inner_loop_size() const;
    long out_loop_size() const;
    virtual void advance_d_ptr();
    virtual void backup_d_ptr();
    long next();

   protected:
    long _numel = 1;
    long _ndim = 0;
    long _out_loop_size = 1;
    long _inner_loop_size = 1;
    long d_ptr = 0;
};

class MultiTensorIterator : public TensorIterator {
   public:
    std::vector<long> shape;
    std::vector<long> shape_m1;
    std::vector<std::vector<long>> strides;
    std::vector<std::vector<long>> strides_back;
    std::vector<std::vector<long>> coordinates;

    MultiTensorIterator(TensorShape& t_shape);

    MultiTensorIterator add_input(TensorShape& t_shape);

    void advance_d_ptr();
    void backup_d_ptr();
    std::vector<long> next();
    long tensor_count = 1;

    std::vector<long> d_ptrs;
    //    private:
};

}  // namespace sail