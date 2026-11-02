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

#ifndef __FDAPDE_EXECUTION_STATIC_DISPATCHER_H__
#define __FDAPDE_EXECUTION_STATIC_DISPATCHER_H__

#include "header_check.h"

namespace fdapde {

// for parallel algorithms requiring the execution of a functor f over a range [begin, end), static partitining divides
// the range [begin, end) in P sub-ranges, and assign a subrange to each worker. subranges size differ by at most one

namespace internals {

// concurrently executes ParallelAlgorithm by dividing the range [begin, end) into P balanced subranges
struct static_dispatcher {
    template <typename ParallelAlgorithm>
        requires(std::is_copy_constructible_v<std::decay_t<ParallelAlgorithm>>) &&
                requires(ParallelAlgorithm alg, int first, int last) {
                    { alg.range() } -> std::same_as<std::pair<int, int>>;
                    { alg.run(first, last) } -> std::same_as<void>;
                }
    static future<void> run(ParallelAlgorithm&& alg) {
        auto* executor = threaded_executor_instance();
        const auto [begin, end] = alg.range();   // query iteration space from algorithm
        if (begin >= end) {                      // nothing to loop on
            auto [future, handle] = make_future<void>();
            handle->set_value();
            handle->complete_task();
            return std::move(future);
        }

        const int distance = end - begin;
        const int n_batches = std::min<int>(distance, executor->size());
        if (n_batches == 1 || executor->size() == 1) {
            // nothing to parallelize, let caller execute the loop serially
            packaged_task<void()> task([&] { alg.run(begin, end); });
            auto future = task.get_future();
            task();
            return future;
        }
        const std::int64_t batch_size = distance / n_batches;
        const int r = distance % n_batches;   // distributes evenly the distance % n_batches iterations
        return task_batch::run(n_batches, [&] {
            return [begin, batch_size, r, alg](const execution_data& data) mutable {
                const int batch_id = data.orig_context;
                // compute task-dependent local range [local_begin, local_end), subranges differ by at most one iter
                const int local_begin = begin + batch_size * batch_id + std::min(batch_id, r);
                const int local_end = local_begin + batch_size + (batch_id < r ? 1 : 0);
                alg.run(local_begin, local_end);
            };
        });
    }
};

}   // namespace internals

// public static dispatching symbol
static constexpr internals::static_dispatcher static_dispatch;

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_STATIC_DISPATCHER_H__
