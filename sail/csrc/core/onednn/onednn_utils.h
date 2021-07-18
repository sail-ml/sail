#pragma once

#include <chrono>
#include <dnnl.hpp>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "Tensor.h"
#include "utils.h"

using namespace std::chrono;

using namespace dnnl;
using dnnl::inner_product_forward;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace sail {

namespace onednn {

template <typename T>
class LRUCache {
   public:
    explicit LRUCache(size_t capacity) {
        capacity_ = capacity;
        clear();
    }

    long size() { return lru_list_.size(); }

    T* get(const std::string& key) {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return nullptr;
        }

        // Move to the front of LRU list as the most recently accessed.
        lru_list_.erase(it->second.lru_iterator);
        lru_list_.push_front(it->first);
        it->second.lru_iterator = lru_list_.begin();
        return it->second.op;
    }

    void add(const std::string& key, T* op) {
        if (lru_list_.size() >= capacity_) {
            _delete();
        }

        // Insert an entry to the front of the LRU list
        lru_list_.push_front(key);
        Item item(op, lru_list_.begin());
        cache_.emplace(std::make_pair(key, std::move(item)));
    }

    void clear() {
        if (lru_list_.empty()) return;

        // Clean up the cache
        cache_.clear();
        lru_list_.clear();
    }

   private:
    struct Item {
        T* op;

        std::list<std::string>::iterator lru_iterator;

        Item(T* op, std::list<std::string>::iterator it) {
            this->op = op;
            this->lru_iterator = it;
        }

        Item(Item&& source) noexcept
            : lru_iterator(std::move(source.lru_iterator)) {
            op = std::move(source.op);
            source.op = std::forward<T*>(nullptr);
        }

        ~Item() {
            if (op != nullptr) delete op;
        }
    };

    // Remove the least recently accessed item from LRU list, which
    // is the tail of lru_list_. Update cache_ correspondingly.
    bool _delete() {
        if (lru_list_.empty()) return false;
        std::string key = lru_list_.back();
        lru_list_.pop_back();
        cache_.erase(key);
        return true;
    }

    // Cache capacity
    size_t capacity_;

    // The cache, a map from std::string key to a LRU item.
    std::unordered_map<std::string, Item> cache_;

    // The LRU list of items.
    std::list<std::string> lru_list_;
};

class Primitive {
   public:
    virtual ~Primitive() = default;
    Primitive() = default;
    Primitive(const engine& cpu_engine) { cpu_engine_ = cpu_engine; }
    // Dummy data which MKL DNN never operates on
    unsigned char* DummyData = nullptr;
    engine cpu_engine_ = engine(engine::kind::cpu, 0);
    const engine& get_engine() { return cpu_engine_; }
};

template <typename T>
class PrimitiveFactory {
   public:
    PrimitiveFactory() = default;

    ~PrimitiveFactory() = default;

    Primitive* get(const std::string& key) {
        auto& lru_cache = PrimitiveFactory<T>::get_cache();
        return lru_cache.get(key);
    }

    void add(const std::string& key, Primitive* op) {
        auto& lru_cache = PrimitiveFactory<T>::get_cache();
        lru_cache.add(key, op);
    }

   private:
    static inline LRUCache<Primitive>& get_cache() {
        static const int kCapacity = 1024;
        static thread_local LRUCache<Primitive> lru_cache_(kCapacity);
        return lru_cache_;
    }
};

class KeyGenerator {
   public:
    KeyGenerator() = default;

    KeyGenerator add(std::string data) {
        append(data);
        return *this;
    }

    KeyGenerator add(int data) {
        auto string = std::to_string(data);
        append(string);
        return *this;
    }

    KeyGenerator add(long data) {
        auto string = std::to_string(data);
        append(string);
        return *this;
    }

    KeyGenerator add(memory::dims dims) {
        for (int i : sail::irange(
                 0, static_cast<int>(
                        dims.size()))) {  // int i = 0; i < dims.size(); i++) {
            add(dims[i]);
        }
        return *this;
    }

    void append(std::string string) {
        key.append(string);
        key.append(1, delim);
    }

    std::string get_key() { return key; }

   private:
    std::string key;
    const char delim = '|';
};

}  // namespace onednn

}  // namespace sail