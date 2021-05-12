#pragma once
#include <boost/atomic.hpp>
#include <boost/intrusive_ptr.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "dtypes.h"
#include "tensor_shape.h"

namespace sail {

// namespace ptr

class TensorBody {
   public:
    typedef boost::intrusive_ptr<TensorBody> pointer;
    // TensorBody() : refcount_(0) {}
    mutable boost::atomic<int> refcount_;

   private:
    friend void intrusive_ptr_add_ref(const TensorBody* x) {
        auto res = x->refcount_.fetch_add(1, boost::memory_order_relaxed);
    }
    friend void intrusive_ptr_release(const TensorBody* x) {
        if (x->refcount_.fetch_sub(1, boost::memory_order_release) == 1) {
            boost::atomic_thread_fence(boost::memory_order_acquire);
            delete x;
        }
    }

   private:
    void* data;
    Dtype dtype;
    TensorShape shape;
    alignemnt_information info;
    bool view;

   public:
    explicit TensorBody(){};

    TensorBody(const TensorBody&) = delete;
    TensorBody& operator=(const TensorBody&) = delete;
    TensorBody(TensorBody&&) = default;
    TensorBody& operator=(TensorBody&&) = default;

    TensorBody(void* _data, Dtype _dtype, TensorShape _shape,
               bool view = false);
    TensorBody(Dtype _dtype, TensorShape _shape, bool view = false);

    ~TensorBody() {
        if (!view) {
            std::free(data);
        }
    }

    void* get_data();
    Dtype get_dtype();
    TensorShape get_shape();
    alignemnt_information get_info();
    bool is_view();

    void set_shape(const TensorShape& s) { shape = s; }
};

}  // namespace sail