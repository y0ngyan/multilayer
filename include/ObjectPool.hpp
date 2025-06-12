#ifndef OBJECT_POOL_HPP
#define OBJECT_POOL_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <stack>
#include <mutex>
#include <cstring>

// 通用对象池模板类
template <typename T>
class ObjectPool {
public:
    // 构造函数，指定初始容量
    explicit ObjectPool(size_t initial_size = 1024) {
        expand(initial_size);
    }

    // 禁掉拷贝构造和赋值
    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;

    // 析构函数
    ~ObjectPool() = default;

    // 从池中获取一个对象（线程安全）
    template <typename... Args>
    T* acquire(Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (free_list_.empty()) {
            size_t current_capacity = 0;
            for(const auto& chunk : pool_) {
                current_capacity += chunk.size();
            }
            expand(current_capacity > 0 ? current_capacity : 1024);
        }

        // 从空闲列表获取
        void* ptr = free_list_.top();
        free_list_.pop();

        // **关键修复**: 在使用 placement new 之前清零内存
        // std::memset(ptr, 0, sizeof(T));
        // **修复**: 只对POD类型清零内存，非POD类型依赖构造函数初始化
        if constexpr (std::is_trivially_constructible_v<T>) {
            std::memset(ptr, 0, sizeof(T));
        }

        // 使用 placement new 构造对象
        T* obj = new (ptr) T(std::forward<Args>(args)...);
        return obj;
    }

    // 将对象归还到池中（线程安全）
    void release(T* obj) {
        if (obj == nullptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // 显式调用对象的析构函数
        obj->~T();

        // **关键修复**: 析构后立即清零内存，确保下次使用时是干净的
        // std::memset(obj, 0, sizeof(T));

        // 将内存地址加回空闲列表
        free_list_.push(static_cast<void*>(obj));
    }

    // 新增：获取池状态信息
    struct PoolStats {
        size_t total_capacity;
        size_t available_count;
        size_t used_count;
        size_t total_chunks;
    };
    
    PoolStats getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        PoolStats stats;
        stats.total_capacity = 0;
        for (const auto& chunk : pool_) {
            stats.total_capacity += chunk.size();
        }
        stats.available_count = free_list_.size();
        stats.used_count = stats.total_capacity - stats.available_count;
        stats.total_chunks = pool_.size();
        return stats;
    }
    
    // 新增：强制清理（用于析构时）
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 清空空闲列表
        while (!free_list_.empty()) {
            free_list_.pop();
        }
        
        // 清空所有内存块
        pool_.clear();
    }

private:
    // 使用简单的字节数组存储
    struct alignas(T) Block {
        char data[sizeof(T)];
    };

    // 存储所有内存块的向量
    std::vector<std::vector<Block>> pool_;
    // 空闲地址栈
    std::stack<void*> free_list_;
    // 线程安全保护
    mutable std::mutex mutex_;

    // 优化扩容策略
    void expand(size_t n) {
        if (n == 0) return;
        
        // **修复**: 限制单次扩容大小，避免内存爆炸
        n = std::min(n, static_cast<size_t>(100000)); // 单次最多扩容10万个对象
        
        try {
            // 创建一个新的内存块
            pool_.emplace_back(n);
            
            // 将所有块地址加入空闲列表
            auto& chunk = pool_.back();
            for (size_t i = 0; i < n; ++i) {
                free_list_.push(static_cast<void*>(&chunk[i]));
            }
            
            std::cout << "[ObjectPool] Expanded with " << n << " objects. Total capacity: " 
                      << getTotalCapacity() << std::endl;
        } catch (const std::bad_alloc& e) {
            std::cerr << "[ObjectPool] ERROR: Failed to allocate memory for " << n 
                      << " objects: " << e.what() << std::endl;
            throw;
        }
    }
    
    size_t getTotalCapacity() const {
        size_t total = 0;
        for (const auto& chunk : pool_) {
            total += chunk.size();
        }
        return total;
    }
};

#endif // OBJECT_POOL_HPP