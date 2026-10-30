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

#ifndef __FDAPDE_EXECUTION_EXECUTION_CONTEXT_H__
#define __FDAPDE_EXECUTION_EXECUTION_CONTEXT_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

static constexpr int external_thread = -1;
inline thread_local int tls_worker_id = external_thread;   // nonnegative IDs identify threads owned by the runtime

// scheduling state for one background worker or the attached caller
struct execution_context {
    using task_pointer = task_handle*;
    using task_queue_type = chase_lev_queue<task_pointer>;
    using task_buffer_type = mpsc_queue<task_pointer>;
    using task_buffer_pointer = typename task_buffer_type::pointer;
    static constexpr int task_queue_size = 8192;

    execution_context(int context_id, bool busy) :
        context_id_(context_id), busy_(busy), task_queue_(task_queue_size) { }

    // attachment/detachment handle
    bool try_attach() noexcept { return !busy_.exchange(true, std::memory_order_acquire); }
    void detach() noexcept { busy_.store(false, std::memory_order_release); }

    // reserve execution capacity for the task
    template <typename Handle> void reserve_task(Handle& handle) {
        handle.task_->set_owner(context_id_);
        if (tls_worker_id == context_id_ && task_queue_.size() + reserved_capacity_ < task_queue_size - 1) {
            // task can be stored in local queue
            ++reserved_capacity_;
            handle.data_ = nullptr;
            return;
        }
	// allocate an unpublished mailbox node
        handle.data_ = task_buffer_.allocate(handle.task_);
	return;
    }
    // release task capacity; local handles (data_ == nullptr) require the context's owning thread
    template <typename Handle> void cancel_task(const Handle& handle) {
        if (handle.data_) {
            task_buffer_.erase(handle.data_);
        } else {
            --reserved_capacity_;
        }
	return;
    }
    // publish reserved tasks; local handles (data_ == nullptr) require the context's owning thread
    template <typename Handle> void publish_task(const Handle& handle) noexcept {
        publish_task(std::addressof(handle), std::addressof(handle) + 1);
    }
    template <std::forward_iterator Iterator> void publish_task(Iterator first, Iterator last) noexcept {
        const auto count = std::ranges::distance(first, last);
        std::size_t mail = 0;
        for (Iterator it = first; it != last; ++it) { mail += (*it).data_ != nullptr; }
        // account for the complete range submission before any task can be executed
        unfinished_tasks_.fetch_add(count, std::memory_order_relaxed);
        ready_tasks_.fetch_add(count, std::memory_order_relaxed);
        if (mail > 0) { incoming_tasks_.fetch_add(mail, std::memory_order_relaxed); }
	// send task for execution
        while (first != last) {
            if ((*first).data_) {
                // store task in buffer if reserved
                task_buffer_.push((*first).data_);
                ++first;
                continue;
            }
	    // search for tasks directly pushable into Chase-Lev queue
            auto end = std::next(first);
            while (end != last && !(*end).data_) { ++end; }
            reserved_capacity_ -= std::distance(first, end);
            auto pointers = std::views::transform(
              std::ranges::subrange(first, end), [](const auto& task) { return task.task_; });
	    // queue tasks
            if (!task_queue_.push_front(pointers.begin(), pointers.end())) std::terminate();
            first = end;
        }
    }
    // signal task completion
    bool complete_task() { return unfinished_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1; }
    // queries context status
    bool has_unfinished_tasks() const { return unfinished_tasks_.load(std::memory_order_acquire) != 0; }
    bool has_ready_tasks() const { return ready_tasks_.load(std::memory_order_acquire) != 0; }

    // perform a work-stealing attempt
    std::optional<task_pointer> try_steal() {
        if (!has_ready_tasks()) { return std::nullopt; }   // idle worker, nothing to steal
        if (auto task = task_queue_.pop_back()) {
            ready_tasks_.fetch_sub(1, std::memory_order_relaxed);
            return task;
        }
        if (incoming_tasks_.load(std::memory_order_relaxed) == 0) { return std::nullopt; }
        std::unique_lock<std::mutex> lock(m_, std::try_to_lock);
        // concurrent stealing attempt, bailing out
        if (!lock.owns_lock()) { return std::nullopt; } 
        auto task = task_buffer_.pop();
        if (task) {   // steal succeded
            incoming_tasks_.fetch_sub(1, std::memory_order_relaxed);
            ready_tasks_.fetch_sub(1, std::memory_order_relaxed);
        }
        return task;
    }
    // only the attached thread may fetch from the local deque
    std::optional<task_pointer> try_fetch_task() {
        if (auto task = task_queue_.pop_front()) {
            ready_tasks_.fetch_sub(1, std::memory_order_relaxed);
            return task;
        }
        if (incoming_tasks_.load(std::memory_order_relaxed) == 0) { return std::nullopt; }
        std::lock_guard<std::mutex> lock(m_);
        auto mail = task_buffer_.pop();
        if (!mail) return std::nullopt;
        std::size_t drained = 1;
        // outstanding local reservations must remain available while draining the mailbox
        while (task_queue_.size() + reserved_capacity_ < task_queue_size - 1) {
            auto task = task_buffer_.pop();
            if (!task) break;
            if (!task_queue_.push_front(*task)) std::terminate();
            ++drained;
        }
        incoming_tasks_.fetch_sub(drained, std::memory_order_relaxed);
        ready_tasks_.fetch_sub(1, std::memory_order_relaxed);
        return mail;
    }
   private:
    const int context_id_;                            // context unique identifier
    std::atomic<bool> busy_;                          // true while a physical thread owns this context
    std::size_t reserved_capacity_ = 0;               // number of reserved queue slots
    task_queue_type task_queue_;                      // local queue of stealable and executable tasks
    task_buffer_type task_buffer_;                    // mailbox accepting submissions from any thread
    std::mutex m_;
  
    std::atomic<std::size_t> unfinished_tasks_ {0};   // submitted tasks not yet completed
    std::atomic<std::size_t> ready_tasks_ {0};        // tasks waiting in the mailbox or local deque
    std::atomic<std::size_t> incoming_tasks_ {0};     // mailbox tasks, including publication in progress
};

}   // namespace internals

// logical identifier of running thread
inline int this_thread_id() noexcept { return internals::tls_worker_id; }

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_EXECUTION_CONTEXT_H__
