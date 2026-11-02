// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_EXECUTION_MPSC_QUEUE_H__
#define __FDAPDE_EXECUTION_MPSC_QUEUE_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// unbounded multi-producer/single-consumer queue; consumption can stall until a producer finishes linking its node
// * "Michael, M. and Scott, M. (1996). Simple, fast, and practical non-blocking and blocking concurrent queue
//    algorithms. In Proceedings of the fifteenth annual ACM symposium on Principles of distributed computing (pp
//    267-275)."
/// @brief owns an unbounded mailbox with multiple producers and one serialized consumer
template <typename T>
    requires(std::is_move_constructible_v<T> && std::is_default_constructible_v<T>)
struct mpsc_queue {
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
   private:
    /// @brief owns a mailbox value and its atomic successor link
    struct node_type {
        /// @brief initializes a mailbox value and its atomic successor link
        node_type() : next(nullptr), data() { }
        /// @brief initializes a mailbox value and its atomic successor link
        template <typename T_>
            requires(std::is_constructible_v<value_type, T_>)
        explicit node_type(T_&& data_) : next(nullptr), data(std::forward<T_>(data_)) { }
        std::atomic<node_type*> next;   // next queue node
        value_type data;                // stored data
    };
   public:
    using allocator_type = pool_allocator<node_type>;
    using pointer = typename allocator_type::pointer;
    using const_pointer = typename allocator_type::const_pointer;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;

    /// @brief creates a mailbox with the default allocator block size
    mpsc_queue() : mpsc_queue(512) { }
    /// @brief creates an empty sentinel mailbox with the supplied allocation block size
    explicit mpsc_queue(size_type allocator_block_sz) : head_(nullptr), tail_(nullptr), allocator_(allocator_block_sz) {
        pointer root = allocator_.allocate();
        head_ = root;
        tail_.store(root, std::memory_order_relaxed);
    }

    // multi-producer push, pushes the element value to the end of the queue
    /// @brief publishes a prepared or newly allocated mailbox node
    template <typename T_>
        requires(std::is_constructible_v<value_type, T_>)
    void push(T_&& value) {
        push(allocate(std::forward<T_>(value)));
    }
    /// @brief publishes a prepared or newly allocated mailbox node
    void push(pointer node) noexcept {
        pointer prev = tail_.exchange(node, std::memory_order_acq_rel);
        prev->next.store(node, std::memory_order_release);
        return;
    }
    // allocate a queue node without pushing it on the queue
    /// @brief reserves a mailbox node without publishing it
    template <typename T_>
        requires(std::is_constructible_v<value_type, T_>)
    pointer allocate(T_&& value) {
        return allocator_.allocate(std::forward<T_>(value));
    }
    // erase an allocated node, node should not be pushed yet
    /// @brief reclaims a reserved mailbox node before it is published
    void erase(pointer node) { allocator_.deallocate(node); }
    // single-consumer pop; returns nullopt if empty or a producer has not yet linked the next node
    /// @brief claims the next linked mailbox value and reclaims the previous sentinel
    std::optional<value_type> pop() {
        pointer next = head_->next.load(std::memory_order_acquire);
        if (next == nullptr) { return std::nullopt; }

        value_type value = std::move(next->data);
        allocator_.deallocate(head_);
        head_ = next;
        return std::move(value);
    }
    /// @brief reclaims the remaining sentinel and nodes after producers and consumers stop
    ~mpsc_queue() {
        while (head_ != nullptr) {
            pointer next = head_->next.load(std::memory_order_relaxed);
            allocator_.deallocate(head_);
            head_ = next;
        }
    }
   private:
    pointer head_;
    alignas(64) std::atomic<pointer> tail_;
    allocator_type allocator_;
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_MPSC_QUEUE_H__
