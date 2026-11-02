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

#ifndef __FDAPDE_EXECUTION_ADAPTIVE_DISPATCHER_H__
#define __FDAPDE_EXECUTION_ADAPTIVE_DISPATCHER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// splits ranges initially to expose work, then refines stolen tasks in response to demand
struct adaptive_dispatcher {
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

        auto [future, handle] = make_future<void>();
        const std::int64_t batch_size = distance / n_batches;
        const int remainder = distance % n_batches;
        try {
            for (int batch_id = 0; batch_id < n_batches; ++batch_id) {
                // compute task-dependent local range [local_begin, local_end), subranges differ by at most one iter
                const int local_begin = begin + batch_size * batch_id + std::min(batch_id, remainder);
                const int local_end = local_begin + batch_size + (batch_id < remainder ? 1 : 0);
                submit_adaptive_task_(alg, local_begin, local_end, max_split_count, handle, batch_id);
            }
        } catch (...) { handle->set_error(std::current_exception()); }
        handle->complete_task();   // retain the producer until all initial tasks have been registered
        return std::move(future);
    }
   private:
    static constexpr int max_split_count = 4;   // number of maximum splits a range can incur
    using future_handle = std::shared_ptr<future<void>::state_type>;

    // specialized task which can spawn a new task when the range is too large, up to split_count times
    template <typename ParallelAlgorithm> struct adaptive_task {
        // constructor
        adaptive_task(const ParallelAlgorithm& alg, const future_handle& handle, int begin, int end, int split_count) :
            alg_(alg), handle_(handle), begin_(begin), end_(end), split_count_(split_count) { }
        // execute the algorithm with additional splitting logic
        void operator()(const execution_data& data) {
            try {
                if (data.exec_context != data.orig_context) {   // check if this task was stolen
                    split_count_ = max_split_count;
                }
                while (split_count_ > 1 && end_ - begin_ > 1) {
                    // half the iteration range and supply a new adaptive task
                    const int middle = begin_ + (end_ - begin_) / 2;
                    const int child_split_count = split_count_ / 2;
                    submit_adaptive_task_(alg_, middle, end_, child_split_count, handle_, data.exec_context);
                    end_ = middle;
                    split_count_ -= child_split_count;
                }
                alg_.run(begin_, end_);
            } catch (...) { handle_->set_error(std::current_exception()); }
            handle_->complete_task();
        }
       private:
        ParallelAlgorithm alg_;
        future_handle handle_;
        int begin_, end_;
        int split_count_;      
    };

    template <typename ParallelAlgorithm>
    static void submit_adaptive_task_(
      const ParallelAlgorithm& alg, int begin, int end, int split_count, const future_handle& handle, int context_id) {
        auto* executor = threaded_executor_instance();
	// spawn task with adaptive range logic
        auto pending_task = adaptive_task<ParallelAlgorithm> {alg, handle, begin, end, split_count};
        auto task = executor->allocate_task(context_id, std::move(pending_task));
        handle->register_task();
	// execute
        try {
            task.submit();
        } catch (...) {
            task.discard();
            handle->complete_task();
            throw;
        }
    }
};

}   // namespace internals

// public adaptive dispatching symbol
static constexpr internals::adaptive_dispatcher adaptive_dispatch;

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_ADAPTIVE_DISPATCHER_H__
