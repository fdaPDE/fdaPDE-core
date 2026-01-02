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

#ifndef __FDAPDE_EXECUTION_TASK_H__
#define __FDAPDE_EXECUTION_TASK_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

class task_handle {
    static constexpr int buffer_size = 64;
   public:
    task_handle() noexcept = default;
    // copy semantic
    task_handle(const task_handle&) = delete;
    task_handle& operator=(const task_handle&) = delete;
    // move semantic
    task_handle(task_handle&& other) noexcept :
        fn_(std::exchange(other.fn_, nullptr)),
        rm_(std::exchange(other.rm_, nullptr)),
        mv_(std::exchange(other.mv_, nullptr)),
        sb_(std::exchange(other.sb_, 0)) {
        if (sb_) {
            if (mv_) { mv_(storage_.buff_, other.storage_.buff_); }
        } else {
            storage_.data_ = std::exchange(other.storage_.data_, nullptr);
        }
	// task dependencies
        parent_    = std::exchange(other.parent_, nullptr);
        ref_count_.store(other.ref_count_);
        other.ref_count_.store(0);
        allocation_pool_ = other.allocation_pool_;   // copy, as physical memory is not moved
    }
    task_handle& operator=(task_handle&& other) noexcept {
        if (this == &other) return *this;
        if (rm_) { rm_(sb_ ? (void*)storage_.buff_ : storage_.data_); }   // clean-up to prevent leaks

        fn_ = std::exchange(other.fn_, nullptr);
        rm_ = std::exchange(other.rm_, nullptr);
        mv_ = std::exchange(other.mv_, nullptr);
        sb_ = std::exchange(other.sb_, 0);
        if (sb_) {
            if (mv_) { mv_(storage_.buff_, other.storage_.buff_); }
        } else {
            storage_.data_ = std::exchange(other.storage_.data_, nullptr);
        }
        // task dependencies
        parent_ = std::exchange(other.parent_, nullptr);
        ref_count_.store(other.ref_count_);
        other.ref_count_.store(0);
        allocation_pool_ = other.allocation_pool_;   // copy, as physical memory is not moved
        return *this;
    }
    template <typename F>
        requires(!std::is_same_v<std::decay_t<F>, task_handle> && std::is_invocable_v<F>)
    explicit task_handle(F&& f, int allocation_pool) :
        parent_(nullptr), ref_count_(0), allocation_pool_(allocation_pool) {
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
    }
    // invoke
    void run() {
        fn_(sb_ ? (void*)storage_.buff_ : storage_.data_);
	// signal task completion, publish writes to successors
        for (task_handle* succ : successors_) { succ->ref_count_.fetch_sub(1, std::memory_order_release); }
    }
    // a task is runnable if all its dependencies have been completed
    bool runnable() const { return ref_count_.load(std::memory_order_acquire) == 0; }
    int ref_count() const { return ref_count_.load(std::memory_order_acquire); }
    int allocation_pool() const { return allocation_pool_; }

    // task dependencies
    template <typename Iterator> void depends_on(Iterator begin, Iterator end) {
        for (Iterator task = begin; task != end; std::advance(task, 1)) {
            (*task)->successors_.push_back(this);
            ref_count_.fetch_add(1, std::memory_order_release);   // number of tasks to wait
        }
	return;
    }
    // destructor (task destroyed only after completion)
    ~task_handle() {
        if (rm_) { rm_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
    }
   private:
    void (*fn_)(void*) = nullptr;
    void (*rm_)(void*) = nullptr;
    void (*mv_)(void*, void*) = nullptr;
    union alignas(std::max_align_t) U {
        void* data_;
        std::byte buff_[buffer_size];   // small buffer optimization
    } storage_ {};
    bool sb_ = false;

    // task properties
    std::vector<task_handle*> successors_ {};   // pointer to successors task
    std::atomic<int> ref_count_ {0};            // number of predecessors tasks not yet completed
    int allocation_pool_ = 0;                   // memory pool physically holding the task
};

}   // namespace internals

// public task API
class Task {
    friend internals::task_handle;
   public:
    Task() : handle_(nullptr) { }
    template <typename F>
        requires(!std::is_same_v<std::decay_t<F>, Task> && std::is_invocable_v<F>)
    explicit Task(F&& f) : handle_(std::make_shared<internals::task_handle>(f)) { }

    template <typename... Tasks> void depends_on(Tasks&&... tasks) {
        std::array<internals::task_handle*, sizeof...(tasks)> handles;
        internals::for_each_index_and_args<sizeof...(tasks)>(
          [&]<int Ns_, typename Task_>(Task_ t) { handles[Ns_] = t.handle().get(); }, tasks...);
        handle_->depends_on(handles.begin(), handles.end());
        return;
    }
    template <typename Iterator> void depends_on(Iterator begin, Iterator end) {
        const int size = std::distance(begin, end);
        std::vector<internals::task_handle*> handles;
        for (Iterator it = begin; it != end; std::advance(it, 1)) { handles.push_back(it->handle_.get()); }
        handle_->depends_on(handles.begin(), handles.end());
        return;
    }
   private:
    std::shared_ptr<internals::task_handle> handle_;
};

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_TASK_H__
