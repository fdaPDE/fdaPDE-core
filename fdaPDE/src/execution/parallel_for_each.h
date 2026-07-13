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

#ifndef __FDAPDE_EXECUTION_PARALLEL_FOR_EACH_H__
#define __FDAPDE_EXECUTION_PARALLEL_FOR_EACH_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// specialized parallelized for-range loop task
struct task_parallel_for_each {
    task_parallel_for_each() = default;

    template <typename Container, typename LoopBody>
        requires(std::is_invocable_v<LoopBody, typename Container::reference>)
    void run(threaded_executor_impl* executor, Container& container, int grain_size, LoopBody&& f) {
        const int n = std::distance(container.begin(), container.end());
        if (n <= 0) return;   // nothing to loop on

        grain_size = std::max(1, std::min(grain_size, n));
        dispatch_(executor, container, grain_size, std::forward<LoopBody>(f));
        return;
    }
    template <typename Container, typename LoopBody>
        requires(std::is_invocable_v<LoopBody, typename Container::reference>)
    void run(threaded_executor_impl* executor, Container& container, LoopBody&& f) {
        const int n = std::distance(container.begin(), container.end());
        if (n <= 0) return;   // nothing to loop on

        int grain_size = std::max(1.0, double(n) / (4 * executor->size()));   // auto-partitioning
        dispatch_(executor, container, grain_size, std::forward<LoopBody>(f));
        return;
    }
   private:
    template <typename Container, typename LoopBody>
        requires(std::is_invocable_v<LoopBody, typename Container::reference>)
    void dispatch_(threaded_executor_impl* executor, Container& container, int grain_size, LoopBody&& f) {
        using iterator_type = std::conditional_t<
          std::is_const_v<Container>, typename std::decay_t<Container>::const_iterator,
          typename std::decay_t<Container>::iterator>;
        iterator_type begin = container.begin();
        iterator_type end = container.end();

        std::atomic<int> local_task_count {1};

        iterator_type chunk_begin = begin;
        while (chunk_begin != end) {
            // register task in the group
            local_task_count.fetch_add(1, std::memory_order_release);

            iterator_type chunk_end =
              std::next(chunk_begin, std::min(grain_size, int(std::distance(chunk_begin, end))));
            auto loop_body = [chunk_begin, chunk_end, &f, &local_task_count]() {
                for (iterator_type it = chunk_begin; it != chunk_end; ++it) { f(*it); }
                // signal task completion
                local_task_count.fetch_sub(1, std::memory_order_release);
            };
            executor->execute(std::move(loop_body));
            // advance the loop
            chunk_begin = chunk_end;
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

// executes for(auto& value : container) { loop_body } in parallel
template <typename Container, typename LoopBody>
    requires(std::is_invocable_v<LoopBody, typename std::decay_t<Container>::reference>)
void parallel_for_each(Container& container, int grain_size, LoopBody&& loop_body) {
    internals::threaded_executor::instance().execute(
      internals::task_parallel_for_each(), container, grain_size, loop_body);
}
template <typename Container, typename LoopBody>
    requires(std::is_invocable_v<LoopBody, typename std::decay_t<Container>::reference>)
void parallel_for_each(Container& container, LoopBody&& loop_body) {
    internals::threaded_executor::instance().execute(internals::task_parallel_for_each(), container, loop_body);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_FOR_EACH_H__
