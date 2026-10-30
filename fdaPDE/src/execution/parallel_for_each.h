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
    template <typename Container, typename LoopBody>
        requires(std::is_invocable_v<LoopBody, typename Container::reference>)
    static void run(threaded_executor_impl* executor, Container& container, int batch_size, LoopBody&& f) {
        const int n = std::distance(container.begin(), container.end());
        if (n <= 0) return;   // nothing to loop on

        batch_size = std::max(1, std::min(batch_size, n));
        const int n_batches = 1 + (n - 1) / batch_size;
        if (n_batches == 1 || executor->size() == 1) {
            for (auto it = container.begin(), end = container.end(); it != end; ++it) { f(*it); }
            return;
        }
        auto local_begin = container.begin();
        int remaining = n;
	task_batch::run(executor, n_batches, [&] {
            const int count = std::min(batch_size, remaining);
            auto local_end = std::next(local_begin, count);
            auto task = [local_begin, local_end, &f] {
                for (auto it = local_begin; it != local_end; ++it) { f(*it); }
            };
            local_begin = local_end;
            remaining -= count;
            return task;
        });
        return;
    }
    template <typename Container, typename LoopBody>
        requires(std::is_invocable_v<LoopBody, typename Container::reference>)
    static void run(threaded_executor_impl* executor, Container& container, LoopBody&& f) {
        const int n = std::distance(container.begin(), container.end());
        if (n <= 0) return;   // nothing to loop on
        // auto-partitioning
        int batch_size = std::max(1.0, double(n) / (4 * executor->size()));
        run(executor, container, batch_size, std::forward<LoopBody>(f));
        return;
    }
};

}   // namespace internals

// executes for(auto& value : container) { loop_body } in parallel
template <typename Container, typename LoopBody>
    requires(std::is_invocable_v<LoopBody, typename std::decay_t<Container>::reference>)
void parallel_for_each(Container& container, int grain_size, LoopBody&& loop_body) {
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for_each::run(executor, container, grain_size, loop_body);
}
template <typename Container, typename LoopBody>
    requires(std::is_invocable_v<LoopBody, typename std::decay_t<Container>::reference>)
void parallel_for_each(Container& container, LoopBody&& loop_body) {
    auto* executor = std::addressof(internals::threaded_executor::instance());
    internals::task_parallel_for_each::run(executor, container, loop_body);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_FOR_EACH_H__
