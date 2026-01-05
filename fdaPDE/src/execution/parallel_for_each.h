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
    task_parallel_for_each() noexcept : m_(), cv_() { }

    template <typename Container, typename LoopBody>
        requires(std::is_invocable_v<LoopBody, typename Container::reference>)
    void run(threaded_executor_impl* executor, const Container& container, LoopBody&& f) {
        using iterator_type = typename Container::iterator_type;
        iterator_type begin = container.begin();
        iterator_type end = container.end();
        const int n = std::distance(begin, end);
        if (n <= 0) return;   // nothing to loop on

        int grain_size = std::max(1.0, double(n) / (4 * executor->size()));
        int local_task_count = 0;
        {
            // lock while dispatching to ensure tasks don't finish and notify before we even finish the loop.
            std::lock_guard<std::mutex> lock(m_);

            iterator_type chunk_begin = begin;
            while (chunk_begin != end) {
                iterator_type chunk_end =
                  std::next(chunk_begin, std::min(grain_size, int(std::distance(chunk_begin, end))));
                local_task_count++;

                auto loop_body = [this, chunk_begin, chunk_end, &f, &local_task_count]() {
                    for (iterator_type it = chunk_begin; it != chunk_end; ++it) { f(*it); }
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
        return;
    }
   private:
    std::mutex m_;
    std::condition_variable cv_;
};

}   // namespace internals

// executes for(auto& value : container) { loop_body } in parallel
template <typename Container, typename LoopBody>
    requires(std::is_invocable_v<LoopBody, typename Container::reference>)
void parallel_for_each(const Container& container, LoopBody&& loop_body) {
    internals::threaded_executor::instance().execute(internals::task_parallel_for_each(), container, loop_body);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_FOR_EACH_H__
