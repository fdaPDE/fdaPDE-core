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

class task_handle {
    static constexpr int buffer_size = 64;
   public:
    task_handle() noexcept = default;
    // copy semantic
    task_handle(const task_handle& other) :
        fn_(other.fn_),
        rm_(other.rm_),
        mv_(other.mv_),
        cp_(other.cp_),
        sb_(other.sb_),
        required_by_(other.required_by_),
        allocation_context_(other.allocation_context_) {
        if (sb_) {
            if (cp_) { cp_(storage_.buff_, other.storage_.buff_); }
        } else {
            storage_.data_ = other.storage_.data_;
        }
        ref_count_.store(other.ref_count_, std::memory_order_release);
    }
    // move semantic
    task_handle(task_handle&& other) noexcept :
        fn_(std::exchange(other.fn_, nullptr)),
        rm_(std::exchange(other.rm_, nullptr)),
        mv_(std::exchange(other.mv_, nullptr)),
        cp_(std::exchange(other.cp_, nullptr)),
        sb_(std::exchange(other.sb_, 0)),
        required_by_(std::move(other.required_by_)),
        allocation_context_(other.allocation_context_)   // copy, as physical memory is not moved
    {
        if (sb_) {
            if (mv_) { mv_(storage_.buff_, other.storage_.buff_); }
        } else {
            storage_.data_ = std::exchange(other.storage_.data_, nullptr);
        }
        // task dependencies
        ref_count_.store(other.ref_count_);
        other.ref_count_.store(0);
    }
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
        required_by_ = std::move(other.required_by_);
        allocation_context_ = other.allocation_context_;
        ref_count_.store(other.ref_count_);
        other.ref_count_.store(0);
        return *this;
    }
    template <typename F, typename AllocationContext>
        requires(!std::is_same_v<std::decay_t<F>, task_handle> && std::is_invocable_v<F> &&
                 std::is_constructible_v<std::optional<int>, AllocationContext>)
    task_handle(F&& f, AllocationContext allocation_context) :
        required_by_(), ref_count_(0), allocation_context_(allocation_context) {
        using Fn = std::decay_t<F>;
        constexpr bool sb = sizeof(F) <= buffer_size && alignof(Fn) <= alignof(union U);
        sb_ = sb;
        if constexpr (sb) {
            // stack allocation for small task object
            new (storage_.buff_) Fn(std::forward<F>(f));
        } else {
            // if task cannot fit in small buffer, resort to heap allocation
            storage_.data_ = new Fn(std::forward<F>(f));
        }
        // type-erased function handlers
        fn_ = [](void* ptr) { (*reinterpret_cast<Fn*>(ptr))(); };
        rm_ = [](void* ptr) noexcept {
            if (ptr == nullptr) { return; }
            (*reinterpret_cast<Fn*>(ptr)).~Fn();
            if constexpr (!sb) { delete (reinterpret_cast<Fn*>(ptr)); }   // free resources
        };
        mv_ = [](void* dst, void* src) noexcept {
            new (dst) Fn(std::move(*reinterpret_cast<Fn*>(src)));
            (*reinterpret_cast<Fn*>(src)).~Fn();
        };
        cp_ = [](void* dst, const void* src) noexcept { new (dst) Fn(*reinterpret_cast<const Fn*>(src)); };
    }
    template <typename F>
        requires(!std::is_same_v<std::decay_t<F>, task_handle> && std::is_invocable_v<F>)
    explicit task_handle(F&& f) : task_handle(std::forward<F>(f), std::nullopt) { }
  
    // invoke
    void run() { fn_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
    // a task is runnable if all its dependencies have been completed (ref_count_ == 0)
    bool runnable() const { return ref_count_.load(std::memory_order_acquire) == 0; }
    int ref_count() const { return ref_count_.load(std::memory_order_acquire); }
    int ref_count_fetch_sub(int i, std::memory_order order = std::memory_order_release) {
        return ref_count_.fetch_sub(i, order);
    }
    int ref_count_fetch_add(int i, std::memory_order order = std::memory_order_release) {
        return ref_count_.fetch_add(i, order);
    }
    std::vector<task_handle*>& required_by() { return required_by_; }
    // memory handling
    const std::optional<int>& allocation_context() const { return allocation_context_; }
    void set_allocation_context(int allocation_context) { allocation_context_ = allocation_context; }
    // destructor (task destroyed only after completion)
    ~task_handle() {
        if (rm_) { rm_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
    }
   private:
    void (*fn_)(void*) = nullptr;
    void (*rm_)(void*) = nullptr;
    void (*mv_)(void*, void*) = nullptr;
    void (*cp_)(void*, const void*) = nullptr;
    union alignas(std::max_align_t) U {
        void* data_;
        std::byte buff_[buffer_size];   // small buffer optimization
    } storage_ {};
    bool sb_ = false;

    // task properties
    std::vector<task_handle*> required_by_ {};   // stable pointers to tasks which require this task to be completed
    std::atomic<int> ref_count_ {0};             // number of not yet completed tasks required by this task
    std::optional<int> allocation_context_;      // memory pool identifier
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_TASK_HANDLE_H__
