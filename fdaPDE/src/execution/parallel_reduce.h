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

#ifndef __FDAPDE_EXECUTION_PARALLEL_REDUCE_H__
#define __FDAPDE_EXECUTION_PARALLEL_REDUCE_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// specialized parallelized reduce task
struct task_parallel_reduce {
    task_parallel_reduce() = default;

    template <typename Iterator, typename T, typename ReduxOp>
        requires(std::is_invocable_r_v<T, ReduxOp, typename Iterator::value_type, typename Iterator::value_type>)
    T run(threaded_executor_impl* executor, Iterator begin, Iterator end, int grain_size, T init, ReduxOp&& redux) {
        const int n = std::distance(begin, end);
        if (n <= 0) return init;   // nothing to loop on

        grain_size = std::max(1, std::min(grain_size, n));
        return dispatch_(executor, begin, end, grain_size, init, std::forward<ReduxOp>(redux));
    }
    template <typename Iterator, typename T, typename ReduxOp>
        requires(std::is_invocable_r_v<T, ReduxOp, typename Iterator::value_type, typename Iterator::value_type>)
    T run(threaded_executor_impl* executor, Iterator begin, Iterator end, T init, ReduxOp&& redux) {
        const int n = std::distance(begin, end);
        if (n <= 0) return init;   // nothing to loop on

        int grain_size = std::max(1.0, double(n) / (4 * executor->size()));
        return dispatch_(executor, begin, end, grain_size, init, std::forward<ReduxOp>(redux));
    }
   private:
    template <typename Iterator, typename T, typename ReduxOp>
    T dispatch_(
      threaded_executor_impl* executor, Iterator begin, Iterator end, int grain_size, T init, ReduxOp&& redux) {
        std::vector<T> partials(executor->size());
        std::atomic<int> local_task_count {1};

        Iterator chunk_begin = begin;
        while (chunk_begin != end) {
            // register task in the group
            local_task_count.fetch_add(1, std::memory_order_release);

            Iterator chunk_end = std::next(chunk_begin, std::min(grain_size, int(std::distance(chunk_begin, end))));
            auto loop_body = [chunk_begin, chunk_end, &redux, &local_task_count, &partials]() {
                T local_init {};
                for (Iterator it = chunk_begin; it != chunk_end; ++it) { local_init = redux(*it, local_init); }
                partials[this_thread_id()] = redux(local_init, partials[this_thread_id()]);
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

        // perform final reduction sequentially
        T v = init;
        for (auto& p : partials) { v = redux(p, v); }
        return v;
    }
};

}   // namespace internals

// general parallel reduction algorithm
template <typename Iterator, typename T, typename ReduxOp>
    requires(std::is_invocable_r_v<T, ReduxOp, typename Iterator::value_type, typename Iterator::value_type>)
T parallel_reduce(Iterator begin, Iterator end, int grain_size, T init, ReduxOp&& redux) {
    return internals::threaded_executor::instance().execute(
      internals::task_parallel_reduce(), begin, end, grain_size, init, redux);
}
template <typename Iterator, typename T, typename ReduxOp>
    requires(std::is_invocable_r_v<T, ReduxOp, typename Iterator::value_type, typename Iterator::value_type>)
T parallel_reduce(Iterator begin, Iterator end, T init, ReduxOp&& redux) {
    return internals::threaded_executor::instance().execute(internals::task_parallel_reduce(), begin, end, init, redux);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_REDUCE_H__
