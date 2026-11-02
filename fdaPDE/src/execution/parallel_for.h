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
template <typename LoopBody> struct task_parallel_for {
    task_parallel_for(int begin, int end, int step, LoopBody f) :
        begin_(begin), end_(end), step_(step), f_(std::move(f)) { }
    // return the iteration space as [0, n_iterations)
    std::pair<int, int> range() const { return {0, begin_ >= end_ ? 0 : 1 + (end_ - begin_ - 1) / step_}; }
    // executes the given iteration range
    void run(int local_begin, int local_end) {
      for (int i = local_begin; i < local_end; ++i) { std::invoke(f_, begin_ + i * step_); }
        return;
    }
   private:
    const int begin_, end_, step_;
    LoopBody f_;
};

}   // namespace internals

// parallel executes for(int i = begin; i < end; i += step) { loop_body(i); }
template <typename LoopBody, typename Dispatcher>
    requires(std::is_invocable_v<std::decay_t<LoopBody>&, int> && std::is_copy_constructible_v<std::decay_t<LoopBody>>)
future<void> parallel_for(int begin, int end, int step, LoopBody&& loop_body, Dispatcher) {
    fdapde_assert(step > 0, "parallel_for step must be positive");
    using task_parallel_for = internals::task_parallel_for<std::decay_t<LoopBody>>;
    return Dispatcher::run(task_parallel_for {begin, end, step, std::forward<LoopBody>(loop_body)});
}

// parallel executes for(int i = begin; i < end; ++i) { loop_body(i); }
  template <typename LoopBody, typename Dispatcher>
    requires(std::is_invocable_v<std::decay_t<LoopBody>&, int> && std::is_copy_constructible_v<std::decay_t<LoopBody>>)
future<void> parallel_for(int begin, int end, LoopBody&& loop_body, Dispatcher d) {
    return parallel_for(begin, end, 1, std::forward<LoopBody>(loop_body), d);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_FOR_H__
