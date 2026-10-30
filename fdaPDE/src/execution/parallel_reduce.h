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
    template <std::forward_iterator Iterator, typename T, typename ReduxOp>
        requires(
          std::convertible_to<std::iter_reference_t<Iterator>, T> &&
          std::is_invocable_r_v<T, ReduxOp&, T, std::iter_reference_t<Iterator>> &&
          std::is_invocable_r_v<T, ReduxOp&, T, T>)
    static T
    run(threaded_executor_impl* executor, Iterator begin, Iterator end, int batch_size, T init, ReduxOp&& redux) {
        const int n = std::distance(begin, end);
        if (n <= 0) return init;   // nothing to loop on

        batch_size = std::max(1, std::min(batch_size, n));
        const int n_batches = 1 + (n - 1) / batch_size;
        if (n_batches == 1 || executor->size() == 1) {
            int remaining = n;
            while (remaining > 0) {
                const int count = std::min(batch_size, remaining);
                auto local_end = std::next(begin, count);
                T local_value(*begin);
                for (++begin; begin != local_end; ++begin) {
                    local_value = std::invoke(redux, std::move(local_value), *begin);
                }
                init = std::invoke(redux, std::move(init), std::move(local_value));
                remaining -= count;
            }
            return init;
        }
        std::vector<std::optional<T>> partials(n_batches);
        Iterator local_begin = begin;
        int batch_index = 0;
        int remaining = n;
        task_batch::run(executor, n_batches, [&] {
            const int count = std::min(batch_size, remaining);
            Iterator local_end = std::next(local_begin, count);
            auto task = [local_begin, local_end, batch_index, &redux, &partials] {
                Iterator it = local_begin;
                T local_value(*it);
                for (++it; it != local_end; ++it) { local_value = std::invoke(redux, std::move(local_value), *it); }
                partials[batch_index].emplace(std::move(local_value));
            };
            local_begin = local_end;
            remaining -= count;
            ++batch_index;
            return task;
        });

        // combine partial results in input order, applying the seed once
        for (auto& partial : partials) { init = std::invoke(redux, std::move(init), std::move(*partial)); }
        return init;
    }
    template <std::forward_iterator Iterator, typename T, typename ReduxOp>
        requires(
          std::convertible_to<std::iter_reference_t<Iterator>, T> &&
          std::is_invocable_r_v<T, ReduxOp&, T, std::iter_reference_t<Iterator>> &&
          std::is_invocable_r_v<T, ReduxOp&, T, T>)
    static T run(threaded_executor_impl* executor, Iterator begin, Iterator end, T init, ReduxOp&& redux) {
        const int n = std::distance(begin, end);
        if (n <= 0) return init;   // nothing to loop on
        // auto-partitioning
        int batch_size = std::max(1.0, double(n) / (4 * executor->size()));
        return run(executor, begin, end, batch_size, init, std::forward<ReduxOp>(redux));
    }
};

}   // namespace internals

// general parallel reduction algorithm
template <std::forward_iterator Iterator, typename T, typename ReduxOp>
    requires(
      std::convertible_to<std::iter_reference_t<Iterator>, T> &&
      std::is_invocable_r_v<T, ReduxOp&, T, std::iter_reference_t<Iterator>> &&
      std::is_invocable_r_v<T, ReduxOp&, T, T>)
T parallel_reduce(Iterator begin, Iterator end, int grain_size, T init, ReduxOp&& redux) {
    auto* executor = std::addressof(internals::threaded_executor::instance());
    return internals::task_parallel_reduce::run(executor, begin, end, grain_size, init, redux);
}
template <std::forward_iterator Iterator, typename T, typename ReduxOp>
    requires(
      std::convertible_to<std::iter_reference_t<Iterator>, T> &&
      std::is_invocable_r_v<T, ReduxOp&, T, std::iter_reference_t<Iterator>> &&
      std::is_invocable_r_v<T, ReduxOp&, T, T>)
T parallel_reduce(Iterator begin, Iterator end, T init, ReduxOp&& redux) {
    auto* executor = std::addressof(internals::threaded_executor::instance());
    return internals::task_parallel_reduce::run(executor, begin, end, init, redux);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_PARALLEL_REDUCE_H__
