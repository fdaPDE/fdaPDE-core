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
template <std::forward_iterator Iterator, typename T, typename ReduxOp> struct task_parallel_reduce {
    struct partial_type {
        int subrange_id;
        T subrange_value;
    };

    task_parallel_reduce(
      Iterator begin, int size, ReduxOp redux, std::vector<partial_type>& partials, std::mutex& partials_mutex) :
        begin_(begin), size_(size), redux_(std::move(redux)), partials_(partials), partials_mutex_(partials_mutex) { }
    // return the iteration space as [0, n_iterations)
    std::pair<int, int> range() const { return {0, size_}; }
    // executes the given iteration range
    void run(int local_begin, int local_end) {
        auto it = std::next(begin_, local_begin);
        // compute local aggregate
        T local_value(*it);
        for (int i = local_begin + 1; i < local_end; ++i) {
            ++it;
            local_value = std::invoke(redux_, std::move(local_value), *it);
        }
        // update partial results
        std::lock_guard<std::mutex> lock(partials_mutex_);
        partials_.push_back({local_begin, std::move(local_value)});
        return;
    }
   private:
    Iterator begin_;
    const int size_;
    ReduxOp redux_;
    std::vector<partial_type>& partials_;
    std::mutex& partials_mutex_;
};

}   // namespace internals

// general parallel reduction algorithm, synchronous execution
template <std::forward_iterator Iterator, typename T, typename ReduxOp, typename Dispatcher>
    requires(
      std::convertible_to<std::iter_reference_t<Iterator>, T> && std::is_invocable_r_v<T, ReduxOp&, T, T> &&
      std::is_copy_constructible_v<std::decay_t<ReduxOp>>)
T parallel_reduce(Iterator begin, Iterator end, T init, ReduxOp&& redux, Dispatcher) {
    const int size = std::distance(begin, end);
    if (size <= 0) return init;   // nothing to loop on
    using task_parallel_reduce = internals::task_parallel_reduce<Iterator, T, std::decay_t<ReduxOp>>;
    // vector of partially aggregated values
    std::vector<typename task_parallel_reduce::partial_type> partials;
    std::mutex partials_mutex;
    auto future = Dispatcher::run(task_parallel_reduce {begin, size, redux, partials, partials_mutex});
    future.get();   // wait for parallel computation

    // final sequential aggregation
    std::sort(partials.begin(), partials.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.subrange_id < rhs.subrange_id;
    });
    // combine partial results in input order, applying the seed once
    for (auto& partial : partials) { init = std::invoke(redux, std::move(init), std::move(partial.subrange_value)); }
    return init;
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_REDUCE_H__
