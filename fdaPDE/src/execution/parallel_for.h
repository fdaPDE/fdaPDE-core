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

// specialized parallelized for loop task
struct task_parallel_for {
    task_parallel_for() = default;

    template <typename LoopBody, typename NextFunctor>
        requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
    void run(threaded_executor_impl* executor, int begin, int end, int grain_size, LoopBody&& f, NextFunctor&& next) {
        int n = 0;
        for (int i = begin; i < end; i = next(i), n++);
        if (n <= 0) return;   // nothing to loop on

        grain_size = std::max(1, std::min(grain_size, n));
        dispatch_(executor, begin, end, grain_size, std::forward<LoopBody>(f), std::forward<NextFunctor>(next));
        return;
    }
    template <typename LoopBody>
        requires(std::is_invocable_v<LoopBody, int>)
    void run(threaded_executor_impl* executor, int begin, int end, int grain_size, LoopBody&& f) {
        const int n = end - begin;
        if (n <= 0) return;   // nothing to loop on

        grain_size = std::max(1, std::min(grain_size, n));
        dispatch_(executor, begin, end, grain_size, std::forward<LoopBody>(f), [](int i) { return ++i; });
        return;
    }
   private:
    template <typename LoopBody, typename NextFunctor>
    void
    dispatch_(threaded_executor_impl* executor, int begin, int end, int grain_size, LoopBody&& f, NextFunctor&& next) {
        std::atomic<int> local_task_count {1};

        for (int j = begin; j < end; j += grain_size) {
            // register task in the group
            local_task_count.fetch_add(1, std::memory_order_release);

            int k = ((end - j) < grain_size) ? end : (j + grain_size);
            auto loop_body = [j, k, next, &f, &local_task_count]() {
                for (int it = j; it < k; it = next(it)) { f(it); }
                // signal task completion
                local_task_count.fetch_sub(1, std::memory_order_release);
            };
            executor->execute(std::move(loop_body));
        }
        // task_group collaborative wait
        local_task_count.fetch_sub(1, std::memory_order_release);
        executor->active_join(this_thread_id(), [&] {
            // help the pool while the task group is not fully consumed
            return local_task_count.load(std::memory_order_acquire) > 0;
        });
        return;
    }
};

}   // namespace internals

// splits [begin, end) into contiguous chunks of size grain_size and submits one task per chunk. Custom loop step logic
template <typename LoopBody, typename NextFunctor>
    requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
void parallel_for(int begin, int end, int grain_size, LoopBody&& loop_body, NextFunctor&& next) {
    internals::threaded_executor::instance().execute(
      internals::task_parallel_for(), begin, end, grain_size, loop_body, next);
}
template <typename LoopBody, typename NextFunctor>
    requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
void parallel_for(int begin, int end, LoopBody&& loop_body, NextFunctor&& next) {
    int grain_size = std::max(1.0, double(end - begin) / (4 * parallel_get_num_threads()));
    internals::threaded_executor::instance().execute(
      internals::task_parallel_for(), begin, end, grain_size, loop_body, next);
}
// splits [begin, end) into contiguous chunks of size grain_size and submits one task per chunk
template <typename LoopBody>
    requires(std::is_invocable_v<LoopBody, int>)
void parallel_for(int begin, int end, int grain_size, LoopBody&& loop_body) {
    internals::threaded_executor::instance().execute(internals::task_parallel_for(), begin, end, grain_size, loop_body);
}
template <typename LoopBody>
    requires(std::is_invocable_v<LoopBody, int>)
void parallel_for(int begin, int end, LoopBody&& loop_body) {
    int grain_size = std::max(1.0, double(end - begin) / (4 * parallel_get_num_threads()));
    internals::threaded_executor::instance().execute(internals::task_parallel_for(), begin, end, grain_size, loop_body);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_FOR_H__
