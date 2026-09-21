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

#ifndef __FDAPDE_EXECUTION_TASK_HANDLE_H__
#define __FDAPDE_EXECUTION_TASK_HANDLE_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

/// @brief owns a copyable type-erased callable with small-buffer storage
class task_handle {
    static constexpr int buffer_size = 64;
   public:
    /// @brief constructs an empty task handle
    task_handle() noexcept = default;

    /// @brief copies owned state independently from the source
    task_handle(const task_handle& other) :
        fn_(other.fn_),
        rm_(other.rm_),
        mv_(other.mv_),
        cp_(other.cp_),
        sb_(other.sb_),
        inverse_deps_(other.inverse_deps_),
        allocation_context_(other.allocation_context_) {
        if (sb_) {
            if (cp_) { cp_(storage_.buff_, other.storage_.buff_); }
        } else {
            if (cp_) { cp_(&storage_.data_, other.storage_.data_); }
        }
        ref_count_.store(other.ref_count_.load(std::memory_order_acquire), std::memory_order_release);
    }
    /// @brief replaces the owned state using the corresponding copy or move operation
    task_handle& operator=(const task_handle& other) {
        if (this == &other) return *this;
        task_handle copy(other);
        *this = std::move(copy);
        return *this;
    }

    /// @brief transfers ownership and leaves the source empty
    task_handle(task_handle&& other) noexcept :
        fn_(std::exchange(other.fn_, nullptr)),
        rm_(std::exchange(other.rm_, nullptr)),
        mv_(std::exchange(other.mv_, nullptr)),
        cp_(std::exchange(other.cp_, nullptr)),
        sb_(std::exchange(other.sb_, 0)),
        inverse_deps_(std::move(other.inverse_deps_)),
        allocation_context_(other.allocation_context_)   // copy, as physical memory is not moved
    {
        if (sb_) {
            if (mv_) { mv_(storage_.buff_, other.storage_.buff_); }
        } else {
            storage_.data_ = std::exchange(other.storage_.data_, nullptr);
        }
        // task dependencies
        ref_count_.store(other.ref_count_.load(std::memory_order_acquire), std::memory_order_release);
        other.ref_count_.store(0);
    }
    /// @brief replaces the owned state using the corresponding copy or move operation
    task_handle& operator=(task_handle&& other) noexcept {
        if (this == &other) return *this;
        if (rm_) { rm_(sb_ ? (void*)storage_.buff_ : storage_.data_); }   // clean-up to prevent leaks

        fn_ = std::exchange(other.fn_, nullptr);
        rm_ = std::exchange(other.rm_, nullptr);
        mv_ = std::exchange(other.mv_, nullptr);
        cp_ = std::exchange(other.cp_, nullptr);
        sb_ = std::exchange(other.sb_, 0);
        if (sb_) {
            if (mv_) { mv_(storage_.buff_, other.storage_.buff_); }
        } else {
            storage_.data_ = std::exchange(other.storage_.data_, nullptr);
        }
        // task dependencies
        inverse_deps_ = std::move(other.inverse_deps_);
        allocation_context_ = other.allocation_context_;
        ref_count_.store(other.ref_count_.load(std::memory_order_acquire), std::memory_order_release);
        other.ref_count_.store(0);
        return *this;
    }
    /// @brief constructs an empty handle or owns the supplied callable
    template <typename F, typename AllocationContext>
        requires(!std::is_same_v<std::decay_t<F>, task_handle> && std::is_invocable_v<std::decay_t<F>&> &&
                 std::is_copy_constructible_v<std::decay_t<F>> &&
                 std::is_constructible_v<std::optional<int>, AllocationContext>)
    task_handle(F&& f, AllocationContext allocation_context) :
        inverse_deps_(), ref_count_(0), allocation_context_(allocation_context) {
        using Fn = std::decay_t<F>;
        constexpr bool sb =
          sizeof(Fn) <= buffer_size && alignof(Fn) <= alignof(union U) && std::is_nothrow_move_constructible_v<Fn>;
        sb_ = sb;
        if constexpr (sb) {   // stack allocation for small task object
            new (storage_.buff_) Fn(std::forward<F>(f));
            rm_ = [](void* ptr) noexcept {
                if (ptr) { reinterpret_cast<Fn*>(ptr)->~Fn(); }
            };
            cp_ = [](void* dst, const void* src) { new (dst) Fn(*reinterpret_cast<const Fn*>(src)); };
        } else {   // if task cannot fit in small buffer, resort to heap allocation
            storage_.data_ = new Fn(std::forward<F>(f));
            rm_ = [](void* ptr) noexcept {
                if (ptr) { delete reinterpret_cast<Fn*>(ptr); }
            };
            cp_ = [](void* dst, const void* src) {
                *reinterpret_cast<void**>(dst) = new Fn(*reinterpret_cast<const Fn*>(src));
            };
        }   // type-erased function handlers
        fn_ = [](void* ptr) { (*reinterpret_cast<Fn*>(ptr))(); };

        mv_ = [](void* dst, void* src) noexcept {
            new (dst) Fn(std::move(*reinterpret_cast<Fn*>(src)));
            (*reinterpret_cast<Fn*>(src)).~Fn();
        };
    }
    /// @brief constructs an empty handle or owns the supplied callable
    template <typename F>
        requires(
          !std::is_same_v<std::decay_t<F>, task_handle> && std::is_invocable_v<std::decay_t<F>&> &&
          std::is_copy_constructible_v<std::decay_t<F>>)
    explicit task_handle(F&& f) : task_handle(std::forward<F>(f), std::nullopt) { }

    /// @brief invokes the owned callable
    void run() { fn_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
    /// @brief reports whether all predecessor tasks have completed
    bool runnable() const { return ref_count_.load(std::memory_order_acquire) == 0; }
    /// @brief loads the pending predecessor count
    int ref_count() const { return ref_count_.load(std::memory_order_acquire); }
    /// @brief decrements the predecessor count and returns its previous value
    int ref_count_fetch_sub(int i, std::memory_order order = std::memory_order_release) {
        return ref_count_.fetch_sub(i, order);
    }
    /// @brief increments the predecessor count and returns its previous value
    int ref_count_fetch_add(int i, std::memory_order order = std::memory_order_release) {
        return ref_count_.fetch_add(i, order);
    }

    /// @brief registers a successor and increments its pending predecessor count
    void inverse_depends_on(task_handle* task) {
        inverse_deps_.push_back(task);
        task->ref_count_fetch_add(1, std::memory_order_relaxed);
    }
    /// @brief returns the successor handles to notify on completion
    const std::vector<task_handle*>& inverse_dependencies() { return inverse_deps_; }

    /// @brief returns the worker pool owning this allocation
    const std::optional<int>& allocation_context() const { return allocation_context_; }
    /// @brief records the worker pool owning this allocation
    void set_allocation_context(int allocation_context) { allocation_context_ = allocation_context; }
    /// @brief releases owned storage and resources
    ~task_handle() {
        if (rm_) { rm_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
    }
   private:
    void (*fn_)(void*) = nullptr;
    void (*rm_)(void*) = nullptr;
    void (*mv_)(void*, void*) = nullptr;
    void (*cp_)(void*, const void*) = nullptr;
    /// @brief stores an inline callable or its owning heap pointer
    union alignas(std::max_align_t) U {
        void* data_ = nullptr;
        std::byte buff_[buffer_size];   // small buffer optimization
    } storage_ {};
    bool sb_ = false;

    // task properties
    std::vector<task_handle*> inverse_deps_ {};   // stable pointers to tasks which require this task to be completed
    std::atomic<int> ref_count_ {0};              // number of not yet completed tasks required by this task
    std::optional<int> allocation_context_;       // memory pool identifier
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_TASK_HANDLE_H__
