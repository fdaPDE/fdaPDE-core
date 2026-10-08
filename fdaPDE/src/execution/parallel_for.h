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

#ifndef __FDAPDE_EXECUTION_PARALLEL_FOR_H__
#define __FDAPDE_EXECUTION_PARALLEL_FOR_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

/// @brief retains loop failures until every published chunk has stopped using its captured state
class parallel_for_task_group {
   public:
    /// @brief keeps the group alive while its submitting caller is still publishing work
    parallel_for_task_group() = default;
    /// @brief registers a chunk before its callable can become visible to workers
    void reserve() { pending_.fetch_add(1, std::memory_order_release); }
    /// @brief releases a completed chunk or a reservation whose submission failed
    void complete() { pending_.fetch_sub(1, std::memory_order_release); }
    /// @brief retains the active exception from the lowest failing iteration index
    void fail(int index) {
        const std::lock_guard lock(mutex_);
        if (index < failure_index_) {
            failure_index_ = index;
            failure_ = std::current_exception();
        }
    }
    /// @brief releases the submitting caller and cooperatively drains every published chunk
    void wait(threaded_executor_impl* executor) {
        complete();
        executor->active_join(this_thread_id(), [&] { return pending_.load(std::memory_order_acquire) > 0; });
    }
    /// @brief rethrows a recorded body failure after all chunks have completed
    void rethrow_failure() const {
        if (failure_) std::rethrow_exception(failure_);
    }
   private:
    std::atomic<int> pending_ {1};
    std::mutex mutex_;
    std::exception_ptr failure_;
    int failure_index_ = std::numeric_limits<int>::max();
};

/// @brief partitions integer iteration ranges into cooperatively joined tasks
struct task_parallel_for {
    /// @brief constructs a stateless parallel task descriptor
    task_parallel_for() = default;

    /// @brief drains all submitted chunks before propagating submission or lowest-index body failures
    template <typename LoopBody>
        requires(std::is_invocable_v<LoopBody, int>)
    void run(threaded_executor_impl* executor, int begin, int end, int grain_size, LoopBody&& f) {
        const int size = end - begin;
        if (size <= 0) return;
        grain_size = std::max(1, std::min(grain_size, size));
        parallel_for_task_group group;
        std::exception_ptr submission_failure;
        try {
            for (int local_begin = begin; local_begin < end;) {
                const int local_end = ((end - local_begin) < grain_size) ? end : (local_begin + grain_size);
                group.reserve();
                try {
                    auto loop_body = [local_begin, local_end, &f, &group]() {
                        int i = local_begin;
                        try {
                            for (; i < local_end; ++i) f(i);
                        } catch (...) { group.fail(i); }
                        group.complete();
                    };
                    executor->execute(std::move(loop_body));
                } catch (...) {
                    // the executor rolls back global accounting before a failed submission reaches this group
                    group.complete();
                    throw;
                }
                local_begin = local_end;
            }
        } catch (...) { submission_failure = std::current_exception(); }
        group.wait(executor);
        if (submission_failure) std::rethrow_exception(submission_failure);
        group.rethrow_failure();
    }
    /// @brief drains stepped chunks before propagating submission, step or lowest-index body failures
    template <typename LoopBody, typename NextFunctor>
        requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
    void run(threaded_executor_impl* executor, int begin, int end, int grain_size, LoopBody&& f, NextFunctor&& next) {
        int size = 0;
        for (int i = begin; i < end; i = next(i), size++);
        if (size <= 0) return;
        grain_size = std::max(1, std::min(grain_size, size));
        const int n_batches = size / grain_size + (size % grain_size != 0);
        parallel_for_task_group group;
        std::exception_ptr submission_failure;
        try {
            int local_begin = begin;
            int local_end = local_begin;
            for (int i = 0; i < grain_size && local_end < end; ++i) local_end = next(local_end);
            for (int j = 0; j < n_batches; ++j) {
                group.reserve();
                try {
                    auto loop_body = [local_begin, local_end, next, &f, &group]() {
                        int i = local_begin;
                        try {
                            for (; i < local_end; i = next(i)) f(i);
                        } catch (...) { group.fail(i); }
                        group.complete();
                    };
                    executor->execute(std::move(loop_body));
                } catch (...) {
                    // retain stack-bound callbacks until every earlier published chunk has completed
                    group.complete();
                    throw;
                }
                local_begin = local_end;
                for (int i = 0; i < grain_size && local_end < end; ++i) local_end = next(local_end);
            }
        } catch (...) { submission_failure = std::current_exception(); }
        group.wait(executor);
        if (submission_failure) std::rethrow_exception(submission_failure);
        group.rethrow_failure();
    }
};

}   // namespace internals

/// @brief executes an integer range in parallel and rethrows failures only after its submitted chunks complete
template <typename LoopBody, typename NextFunctor>
    requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
void parallel_for(int begin, int end, int grain_size, LoopBody&& loop_body, NextFunctor&& next) {
    internals::threaded_executor::instance().execute(
      internals::task_parallel_for(), begin, end, grain_size, loop_body, next);
}
/// @brief executes an integer range in parallel and rethrows failures only after its submitted chunks complete
template <typename LoopBody, typename NextFunctor>
    requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
void parallel_for(int begin, int end, LoopBody&& loop_body, NextFunctor&& next) {
    int grain_size = std::max(1.0, double(end - begin) / (4 * parallel_get_num_threads()));
    internals::threaded_executor::instance().execute(
      internals::task_parallel_for(), begin, end, grain_size, loop_body, next);
}
/// @brief executes an integer range in parallel and rethrows failures only after its submitted chunks complete
template <typename LoopBody>
    requires(std::is_invocable_v<LoopBody, int>)
void parallel_for(int begin, int end, int grain_size, LoopBody&& loop_body) {
    internals::threaded_executor::instance().execute(internals::task_parallel_for(), begin, end, grain_size, loop_body);
}
/// @brief executes an integer range in parallel and rethrows failures only after its submitted chunks complete
template <typename LoopBody>
    requires(std::is_invocable_v<LoopBody, int>)
void parallel_for(int begin, int end, LoopBody&& loop_body) {
    int grain_size = std::max(1.0, double(end - begin) / (4 * parallel_get_num_threads()));
    internals::threaded_executor::instance().execute(internals::task_parallel_for(), begin, end, grain_size, loop_body);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_FOR_H__
