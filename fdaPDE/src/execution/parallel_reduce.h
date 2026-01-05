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
    task_parallel_reduce() noexcept : m_(), cv_() { }

    template <typename Iterator, typename T, typename ReduxOp>
    T run(threaded_executor_impl* executor, Iterator begin, Iterator end, T init, ReduxOp&& redux) {
        const int n = std::distance(begin, end);
        if (n <= 0) return init;   // nothing to loop on

        std::vector<T> partials(executor->size());

        int grain_size = std::max(1.0, double(n) / (4 * executor->size()));
        int local_task_count = 0;
        {
            // lock while dispatching to ensure tasks don't finish and notify before we even finish the loop.
            std::lock_guard<std::mutex> lock(m_);

            Iterator chunk_begin = begin;
            while (chunk_begin != end) {
                Iterator chunk_end = std::next(chunk_begin, std::min(grain_size, int(std::distance(chunk_begin, end))));
                local_task_count++;

                auto loop_body = [this, chunk_begin, chunk_end, &redux, &local_task_count, &partials]() {
                    T local_init {};
                    for (Iterator it = chunk_begin; it != chunk_end; ++it) { local_init = redux(*it, local_init); }
                    partials[this_worker_id() - 1] = redux(local_init, partials[this_worker_id() - 1]);
                    {
                        std::lock_guard<std::mutex> lock(this->m_);
                        local_task_count--;
                        if (local_task_count == 0) { this->cv_.notify_all(); }
                    }
                };
                executor->execute(std::move(loop_body));
                // advance the loop
                chunk_begin = chunk_end;
            }
        }
        // wait until all dispatched tasks are complete
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return local_task_count == 0; });

        // perform final reduction sequentially
        T v = init;
        for (auto& p : partials) { v = redux(p, v); }
        return v;
    }
   private:
    std::mutex m_;
    std::condition_variable cv_;
};

}   // namespace internals

// general parallel reduction algorithm
template <typename Iterator, typename T, typename ReduxOp>
T parallel_reduce(Iterator begin, Iterator end, T init, ReduxOp&& redux) {
    return internals::threaded_executor::instance().execute(internals::task_parallel_reduce(), begin, end, init, redux);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_REDUCE_H__
