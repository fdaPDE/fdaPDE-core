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
    /// @brief starts without a recorded chunk failure
    parallel_for_task_group() = default;
    /// @brief retains the active exception from the lowest failing iteration index
    void fail(int index) {
        const std::lock_guard lock(mutex_);
        if (index < failure_index_) {
            failure_index_ = index;
            failure_ = std::current_exception();
        }
    }
    /// @brief rethrows a recorded body failure after all chunks have completed
    void rethrow_failure() const {
        if (failure_) std::rethrow_exception(failure_);
    }
   private:
    std::mutex mutex_;
    std::exception_ptr failure_;
    int failure_index_ = std::numeric_limits<int>::max();
};

/// @brief partitions integer ranges into collaboratively joined batches
struct task_parallel_for {
    /// @brief partitions integer iterations and drains chunks before rethrowing the lowest-index failure
    template <typename LoopBody>
        requires(std::is_invocable_v<LoopBody, int>)
    static void run(threaded_executor_impl* executor, int begin, int end, int batch_size, LoopBody&& f) {
        if (end <= begin) return;
        batch_size = std::max(1, batch_size);
        const int distance = end - begin;

        batch_size = std::min<int>(batch_size, distance);
        const int n_batches = 1 + (distance - 1) / batch_size;
        if (n_batches == 1 || executor->size() == 1) {
            // nothing to parallelize, let caller execute the loop serially
            for (int i = begin; i < end; ++i) { f(i); }
            return;
        }
        parallel_for_task_group failures;
        int local_begin = begin;
        task_batch::run(executor, n_batches, [&] {
            const int local_end = end - local_begin < batch_size ? end : local_begin + batch_size;
            auto task = [local_begin, local_end, &f, &failures]() mutable {
                int i = local_begin;
                try {
                    for (; i < local_end; ++i) f(i);
                } catch (...) { failures.fail(i); }
            };
            local_begin = local_end;
            return task;
        });
        failures.rethrow_failure();
        return;
    }

    /// @brief partitions integer iterations and drains chunks before rethrowing the lowest-index failure
    template <typename LoopBody, typename NextFunctor>
        requires(
          std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int> &&
          std::is_copy_constructible_v<std::decay_t<NextFunctor>>)
    static void
    run(threaded_executor_impl* executor, int begin, int end, int batch_size, LoopBody&& f, NextFunctor&& next) {
        if (end <= begin) return;
        batch_size = std::max(1, batch_size);
        // count number of hops according to next functor
        int size = 0;
        for (int i = begin; i < end; i = next(i), ++size);
        if (size <= 0) { return; }   // nothing to loop on

        batch_size = std::max(1, std::min(batch_size, size));
        const int n_batches = 1 + (size - 1) / batch_size;
        if (n_batches == 1 || executor->size() == 1) {
            // nothing to parallelize, let caller execute the loop serially
            for (int i = begin; i < end; i = next(i)) { f(i); }
            return;
        }
        parallel_for_task_group failures;
        int local_begin = begin;
        task_batch::run(executor, n_batches, [&] {
            int local_end = local_begin;
            for (int i = 0; i < batch_size && local_end < end; ++i) { local_end = next(local_end); }
            auto task = [local_begin, local_end, next, &f, &failures]() mutable {
                int i = local_begin;
                try {
                    for (; i < local_end; i = next(i)) f(i);
                } catch (...) { failures.fail(i); }
            };
            local_begin = local_end;
            return task;
        });
        failures.rethrow_failure();
        return;
    }
};

}   // namespace internals

// splits [begin, end) into contiguous chunks of size grain_size and submits one task per chunk. Custom loop step logic
/// @brief executes an integer range and reports the lowest-index failure after its chunks finish
template <typename LoopBody, typename NextFunctor>
    requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
void parallel_for(int begin, int end, int grain_size, LoopBody&& loop_body, NextFunctor&& next) {
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for::run(executor, begin, end, grain_size, loop_body, next);
}
/// @brief executes an integer range and reports the lowest-index failure after its chunks finish
template <typename LoopBody, typename NextFunctor>
    requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
void parallel_for(int begin, int end, LoopBody&& loop_body, NextFunctor&& next) {
    int batch_size = std::max(1.0, double(end - begin) / (4 * parallel_get_num_threads()));
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for::run(executor, begin, end, batch_size, loop_body, next);
}
// splits [begin, end) into contiguous chunks of size batch_size and submits one task per chunk
/// @brief executes an integer range and reports the lowest-index failure after its chunks finish
template <typename LoopBody>
    requires(std::is_invocable_v<LoopBody, int>)
void parallel_for(int begin, int end, int batch_size, LoopBody&& loop_body) {
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for::run(executor, begin, end, batch_size, loop_body);
}
/// @brief executes an integer range and reports the lowest-index failure after its chunks finish
template <typename LoopBody>
    requires(std::is_invocable_v<LoopBody, int>)
void parallel_for(int begin, int end, LoopBody&& loop_body) {
    int batch_size = std::max(1.0, double(end - begin) / (4 * parallel_get_num_threads()));
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for::run(executor, begin, end, batch_size, loop_body);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_FOR_H__
