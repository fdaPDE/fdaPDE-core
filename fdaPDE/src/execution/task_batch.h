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

// submits a batch of tasks and returns its completion future
struct task_batch {
    template <typename TaskGenerator>
    static future<void> run(int n_batches, TaskGenerator&& generator) {
        fdapde_assert(n_batches >= 0);
        using task_submission_handle = typename threaded_executor::task_submission_handle;
        threaded_executor* executor = threaded_executor_instance();
        auto [future, handle] = make_future<void>();
        std::vector<task_submission_handle> task_group;
        // batches are sent at slots of size slot_size to lower synchronization overhead
        const int slot_size = std::min<int>(n_batches, 4 * executor->size());
        std::size_t published = 0;
        try {
            task_group.reserve(slot_size);
            for (std::int64_t i = 0; i < n_batches; ++i) {
                auto loop_body = std::invoke(generator);
                auto task = [handle, loop_body = std::move(loop_body)](const execution_data& data) mutable {
                    try {
                        invoke_task(loop_body, data);
                    } catch (...) {
                        handle->set_error(std::current_exception());
                    }
                    handle->complete_task();
                };
                // distribute tasks in round-robin order across the workers
                task_group.push_back(executor->allocate_task(i % executor->size(), std::move(task)));
                handle->register_task();
                // send tasks to execution
                if (std::cmp_equal(task_group.size(), slot_size) || i + 1 == n_batches) {
                    // publication precedes notification, which can throw after all tasks become visible
                    published = task_group.size();
                    executor->submit_task(task_group);
                    task_group.clear();
                    published = 0;
                }
            }
            handle->set_value();
        } catch (...) {
            handle->set_error(std::current_exception());
            // cancel unpublished tasks
            for (std::size_t i = published; i < task_group.size(); ++i) {
                task_group[i]->discard();
                handle->complete_task();
            }
        }
        handle->complete_task();   // release the producer's initial reservation
        return std::move(future);
    }
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_TASK_BATCH_H__
