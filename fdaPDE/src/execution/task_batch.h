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

#ifndef __FDAPDE_EXECUTION_TASK_BATCH_H__
#define __FDAPDE_EXECUTION_TASK_BATCH_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// executes a batch of tasks and wait for completion
struct task_batch {
    template <typename TaskGenerator>
    static void run(threaded_executor_impl* executor, int n_batches, TaskGenerator&& generator) {
        using task_submission_handle = typename threaded_executor_impl::task_submission_handle;
        std::vector<task_submission_handle> task_group;
        // batches are sent at slots of size slot_size to lower synchronization overhead
        const int slot_size = std::min<int>(n_batches, 4 * executor->size());
        task_group.reserve(slot_size);
        // error handling
        std::mutex m;
        std::exception_ptr error;
        // calling worker tries task-helping. pool detachment is executed at handle destruction
        auto attachment_handle = executor->try_attach();

        std::atomic<int> task_group_count {n_batches - 1};
        std::size_t submitted = 0;
        try {
            for (std::int64_t i = 0; i < n_batches - 1; ++i) {
                auto loop_body = std::invoke(generator);
                auto task = [&, executor, loop_body = std::optional {std::move(loop_body)}]() mutable {
                    try {
                        std::invoke(*loop_body);   // execute loop body
                    } catch (...) {
                        // store error
                        std::lock_guard<std::mutex> lock(m);
                        if (!error) { error = std::current_exception(); }
                    }
                    loop_body.reset();   // release captures before reporting completion
                    // signal task group completion
                    if (task_group_count.fetch_sub(1, std::memory_order_acq_rel) == 1) { executor->notify_all_join(); }
                };
                task_group.push_back(executor->allocate_task(this_thread_id(), std::move(task)));
		// send tasks to execution
                if (std::cmp_equal(task_group.size(), slot_size)) {
                    submitted += task_group.size();
                    executor->submit_task(task_group);
                    task_group.clear();
                }
            }
            if (task_group.size() > 0) {
                submitted += task_group.size();
                executor->submit_task(task_group);
                task_group.clear();
            }
            std::invoke(generator)();   // execute the final chunk directly
        } catch (...) {
            // un-published tasks are canceled. already published tasks must drain before rethrowing.
            for (auto& task : task_group) { task->discard(); }
            // store error
            std::lock_guard<std::mutex> lock(m);
            if (!error) { error = std::current_exception(); }
        }
        if (std::cmp_not_equal(submitted, n_batches - 1)) {
            task_group_count.fetch_sub((n_batches - 1) - submitted, std::memory_order_acq_rel);
        }
        // all published work must finish before local state dies
        executor->active_join(this_thread_id(), [&] { return task_group_count.load(std::memory_order_acquire) != 0; });
        if (error) { std::rethrow_exception(error); }
	return;
    }
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_TASK_BATCH_H__
