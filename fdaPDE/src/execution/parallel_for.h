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
    template <typename LoopBody>
        requires(std::is_invocable_v<LoopBody, int>)
    static void run(threaded_executor_impl* executor, int begin, int end, int batch_size, LoopBody&& f) {
        fdapde_assert(begin <= end && batch_size > 0);
        const int distance = end - begin;
        if (distance <= 0) { return; }   // nothing to loop on

        batch_size = std::min<int>(batch_size, distance);
        const int n_batches = 1 + (distance - 1) / batch_size;
        if (n_batches == 1 || executor->size() == 1) {
            // nothing to parallelize, let caller execute the loop serially
            for (int i = begin; i < end; ++i) { f(i); }
            return;
        }
        int local_begin = begin;
        task_batch::run(executor, n_batches, [&] {
            const int local_end = std::min<int>(end, local_begin + batch_size);
            auto task = [local_begin, local_end, &f]() mutable {
                for (int i = local_begin; i < local_end; ++i) { f(i); }
            };
            local_begin = local_end;
            return task;
        });
        return;
    }

    // custom steps parallel_for loop
    template <typename LoopBody, typename NextFunctor>
        requires(
          std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int> &&
          std::is_copy_constructible_v<std::decay_t<NextFunctor>>)
    static void
    run(threaded_executor_impl* executor, int begin, int end, int batch_size, LoopBody&& f, NextFunctor&& next) {
        fdapde_assert(begin <= end && batch_size > 0);
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
        int local_begin = begin;
	task_batch::run(executor, n_batches, [&] {
            int local_end = local_begin;
            for (int i = 0; i < batch_size && local_end < end; ++i) { local_end = next(local_end); }
            auto task = [local_begin, local_end, next, &f]() mutable {
                for (int it = local_begin; it < local_end; it = next(it)) { f(it); }
            };
            local_begin = local_end;
            return task;
        });
	return;
    }
};

}   // namespace internals

// splits [begin, end) into contiguous chunks of size grain_size and submits one task per chunk. Custom loop step logic
template <typename LoopBody, typename NextFunctor>
    requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
void parallel_for(int begin, int end, int grain_size, LoopBody&& loop_body, NextFunctor&& next) {
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for::run(executor, begin, end, grain_size, loop_body, next);
}
template <typename LoopBody, typename NextFunctor>
    requires(std::is_invocable_v<LoopBody, int> && std::is_invocable_r_v<int, NextFunctor, int>)
void parallel_for(int begin, int end, LoopBody&& loop_body, NextFunctor&& next) {
    int batch_size = std::max(1.0, double(end - begin) / (4 * parallel_get_num_threads()));
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for::run(executor, begin, end, batch_size, loop_body, next);
}
// splits [begin, end) into contiguous chunks of size batch_size and submits one task per chunk
template <typename LoopBody>
    requires(std::is_invocable_v<LoopBody, int>)
void parallel_for(int begin, int end, int batch_size, LoopBody&& loop_body) {
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for::run(executor, begin, end, batch_size, loop_body);
}
template <typename LoopBody>
    requires(std::is_invocable_v<LoopBody, int>)
void parallel_for(int begin, int end, LoopBody&& loop_body) {
    int batch_size = std::max(1.0, double(end - begin) / (4 * parallel_get_num_threads()));
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for::run(executor, begin, end, batch_size, loop_body);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_FOR_H__
