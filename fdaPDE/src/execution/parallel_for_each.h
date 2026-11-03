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
template <std::forward_iterator Iterator, typename LoopBody> struct task_parallel_for_each {
    task_parallel_for_each(Iterator begin, int size, LoopBody f) : begin_(begin), size_(size), f_(std::move(f)) { }
    // return the iteration space as [0, n_iterations)
    std::pair<int, int> range() const { return {0, size_}; }
    // executes the given iteration range
    void run(int local_begin, int local_end) {
        auto it = std::next(begin_, local_begin);
        for (int i = local_begin; i < local_end; ++i, ++it) { std::invoke(f_, *it); }
	return;
    }
   private:
    Iterator begin_;
    const int size_;
    LoopBody f_;
};
  
}   // namespace internals

// parallel executes for(auto& value : container) { loop_body(value); } 
template <std::ranges::forward_range Container, typename LoopBody, typename Dispatcher>
    requires(
      std::is_invocable_v<std::decay_t<LoopBody>&, std::ranges::range_reference_t<Container>> &&
      std::is_copy_constructible_v<std::decay_t<LoopBody>>)
future<void> parallel_for_each(Container& container, LoopBody&& loop_body, Dispatcher) {
    using task_parallel_for_each =
      internals::task_parallel_for_each<std::ranges::iterator_t<Container>, std::decay_t<LoopBody>>;
    const int size = std::ranges::distance(container);
    return Dispatcher::run(task_parallel_for_each {std::ranges::begin(container), size, std::forward<LoopBody>(loop_body)});
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_FOR_EACH_H__
