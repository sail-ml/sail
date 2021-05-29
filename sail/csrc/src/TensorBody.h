#pragma once
#include <immintrin.h>
// #include <boost/atomic.hpp>
#include <boost/intrusive_ptr.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <atomic>
#include "dtypes.h"
#include "tensor_shape.h"
#include "utils.h"

namespace sail {

// namespace ptr

class Tensor;

class TensorBody {
   public:
    typedef boost::intrusive_ptr<TensorBody> pointer;
    // TensorBody() : refcount_(0) {}
    mutable std::atomic<int> refcount_;
    // mutable boost::atomic<int> refcount_;

   private:
    friend void intrusive_ptr_add_ref(const TensorBody* x) {
        auto res = x->refcount_.fetch_add(1, std::memory_order_relaxed);
    }
    friend void intrusive_ptr_release(const TensorBody* x) {
        if (x->refcount_.fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            delete x;
        }
    }

   private:
    void* data = NULL;
    Dtype dtype;
    TensorShape* shape = NULL;
    alignemnt_information info;
    bool view = false;
    bool _has_grad = false;
    Tensor* grad = NULL;
    ;

   public:
    explicit TensorBody(){};

    TensorBody(const TensorBody&) = delete;
    TensorBody& operator=(const TensorBody&) = delete;
    TensorBody(TensorBody&&) = default;
    TensorBody& operator=(TensorBody&&) = default;

    TensorBody(void* _data, Dtype _dtype, TensorShape _shape,
               bool view = false);
    TensorBody(void*& _data, Dtype& _dtype, TensorShape& _shape, bool& view);

    TensorBody(Dtype _dtype, TensorShape _shape, bool view = false);

    ~TensorBody();

    TensorBody::pointer create_owner();

    inline void* get_data() { return data; }
    inline Dtype get_dtype() { return dtype; }
    inline TensorShape get_shape() { return *shape; }
    inline alignemnt_information get_info() { return info; }
    inline bool is_view() { return view; }
    inline bool has_grad() { return _has_grad; }

    Tensor get_grad();
    void set_grad(Tensor& t);

    inline int get_ref_count() { return (int)refcount_; }

    void force_incref() { refcount_.fetch_add(1, std::memory_order_relaxed); }

    void set_shape(const TensorShape& s) {
        delete shape;
        shape = new TensorShape(s);
        // shape = s;
    }

    long int* get_shape_ptr() { return shape->get_shape_ptr(); }
};

}  // namespace sail