#ifndef OBJECT_POOL_HPP
#define OBJECT_POOL_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <type_traits>

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

    // 析构函数，确保所有对象都被正确析构
    ~ObjectPool() {
        // 注意：这里不需要手动释放对象，因为我们假设用户会正确release
        // 如果需要更严格的管理，可以添加活跃对象追踪
    }

    // 从池中获取一个对象
    template <typename... Args>
    T* acquire(Args&&... args) {
        if (free_list_ == nullptr) {
            size_t current_capacity = 0;
            for(const auto& chunk : pool_) {
                current_capacity += chunk.size();
            }
            expand(current_capacity > 0 ? current_capacity : 1024);
        }

        // 从空闲列表头部取出一个节点
        Node* head = free_list_;
        free_list_ = head->next;

        // 使用 placement new 在已分配的内存上构造对象
        T* obj = new (&head->storage) T(std::forward<Args>(args)...);
        return obj;
    }

    // 将对象归还到池中
    void release(T* obj) {
        if (obj == nullptr) {
            return;
        }

        // 显式调用对象的析构函数
        obj->~T();

        // 将对象的内存重新解释为 Node 指针，并将其插入到空闲列表的头部
        // Node* head = reinterpret_cast<Node*>(obj);
        Node* head = new (obj) Node();
        head->next = free_list_;
        free_list_ = head;
    }

private:
    // 修复内存对齐问题
    struct Node {
        alignas(T) char storage[sizeof(T)];
        Node* next;
        
        Node() : next(nullptr) {}
    };

    
    // 存储所有内存块的向量
    std::vector<std::vector<Node>> pool_;
    // 指向空闲列表的头指针
    Node* free_list_ = nullptr;

    // 扩容函数
    void expand(size_t n) {
        if (n == 0) return;
        
        // 创建一个新的内存块
        pool_.emplace_back(n);
        
        // 将新内存块中的所有节点串联起来，形成一个链表
        Node* chunk = pool_.back().data();
        
        // 将这个新的链表连接到现有空闲列表的前面
        if (n > 0) {
            chunk[n - 1].next = free_list_;
            for (size_t i = 0; i < n - 1; ++i) {
                chunk[i].next = &chunk[i + 1];
            }
            free_list_ = &chunk[0];
        }
    }
};

#endif // OBJECT_POOL_HPP