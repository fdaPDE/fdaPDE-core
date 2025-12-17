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

#ifndef __FDAPDE_GRID_SEARCH_H__
#define __FDAPDE_GRID_SEARCH_H__

#include "header_check.h"

namespace fdapde {

template <int N> class GridSearch {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;

    vector_t optimum_;
    double value_;                 // objective value at optimum
    std::vector<double> values_;   // explored objective values during optimization
    int size_;
   public:
    static constexpr bool gradient_free = true;
    static constexpr int static_input_size = N;
    vector_t x_curr;
    double obj_curr;
    // constructor
    GridSearch() : size_(N) { fdapde_static_assert(N != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_GRID_SEARCH_ONLY); }
    GridSearch(int size) : size_(N == Dynamic ? size : N) { fdapde_assert(N == Dynamic || size == N); }
    GridSearch(const GridSearch& other) : size_(other.size_) { }
    GridSearch& operator=(const GridSearch& other) {
        size_ = other.size_;
        return *this;
    }
    template <typename ObjectiveT, typename GridT, typename... Callbacks>
        requires((internals::is_vector_like_v<GridT> || internals::is_matrix_like_v<GridT>))
    vector_t optimize(ObjectiveT&& objective, const GridT& grid, Callbacks&&... callbacks) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE);
        using layout_policy = decltype([]() {
            if constexpr (internals::is_eigen_dense_xpr_v<GridT>) {
                return std::conditional_t<GridT::IsRowMajor, internals::layout_right, internals::layout_left> {};
            } else {
                return internals::layout_right {};
            }
        }());
        using grid_t = MdMap<const double, MdExtents<Dynamic, Dynamic>, layout_policy>;
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

        std::tuple<Callbacks...> callbacks_ {callbacks...};
        grid_t grid_;
        value_ = std::numeric_limits<double>::max();
        if constexpr (internals::is_vector_like_v<GridT>) {
            fdapde_assert(grid.size() % size_ == 0);
            grid_ = grid_t(grid.data(), grid.size() / size_, size_);
        } else {
            fdapde_assert(grid.cols() == size_);
            grid_ = grid_t(grid.data(), grid.rows(), size_);
        }
        bool stop = false;   // asserted true in case of forced stop
        grid_.row(0).assign_to(x_curr.transpose());
        obj_curr = objective(x_curr);
        stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
        values_.clear();
        values_.push_back(obj_curr);
        if (obj_curr < value_) {
            value_ = obj_curr;
            optimum_ = x_curr;
        }
        // optimize field over supplied grid
        for (std::size_t i = 1; i < grid_.rows() && !stop; ++i) {
            grid_.row(i).assign_to(x_curr.transpose());
            obj_curr = objective(x_curr);
            stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
            values_.push_back(obj_curr);
            // update minimum if better optimum found
            if (obj_curr < value_) {
                value_ = obj_curr;
                optimum_ = x_curr;
            }
            stop |= internals::exec_stop_if(*this, objective);
        }
        return optimum_;
    }

    // Parallel optimize overload: each worker of the threadpool searches in a fraction of the grid, then a final
    // sequential reduction is performed on the workers' optimal results
    template <typename ObjectiveT, typename GridT, typename Threadpool>
        requires((internals::is_vector_like_v<GridT> || internals::is_matrix_like_v<GridT>))
    vector_t optimize(
      ObjectiveT&& objective, const GridT& grid, execution::execution_parallel, Threadpool& Tp,
      int granularity =
        -1) {   // per ora int job_per_worker in input perche piu comodo fare i test poi sostituire valore scelto
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE);
        using layout_policy = decltype([]() {
            if constexpr (internals::is_eigen_dense_xpr_v<GridT>) {
                return std::conditional_t<GridT::IsRowMajor, internals::layout_right, internals::layout_left> {};
            } else {
                return internals::layout_right {};
            }
        }());
        using grid_t = MdMap<const double, MdExtents<Dynamic, Dynamic>, layout_policy>;

        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

        grid_t grid_;
        value_ = std::numeric_limits<double>::max();
        if constexpr (internals::is_vector_like_v<GridT>) {
            fdapde_assert(grid.size() % size_ == 0);
            grid_ = grid_t(grid.data(), grid.size() / size_, size_);
        } else {
            fdapde_assert(grid.cols() == size_);
            grid_ = grid_t(grid.data(), grid.rows(), size_);
        }

        // to avoid false-sharing
        struct alignas(64) AlignedPair {
            double first =
              std::numeric_limits<double>::max();   // inizializzato a massimo erch ein problema cerchiamo minimo
            vector_t second;
        };
        int n_threads = Tp.n_workers();
        // Vector of (value, optimum) for each worker. At the end, it contains the min/argmin found by each worker, and
        // reducing this vector yields the final min/argmin. Initialized with n_threads empty elements to avoid
        // reallocations and ensure thread safety
        std::vector<AlignedPair> value_optimum_workers(n_threads);

        // If default (input = -1), set granularity so that each worker receives 1 job (+1 extra job for the remainder,
        // but with fewer than n_threads iterations, so negligible)
        if (granularity == -1) { granularity = (grid_.rows() / n_threads > 0) ? grid_.rows() / n_threads : 1; }
        int granularity_last_job = grid_.rows() % granularity;
        int n_job = (granularity_last_job == 0) ? grid_.rows() / granularity : grid_.rows() / granularity + 1;

        Tp.parallel_for(0, n_job, [&, this](int i) {
            int index_worker = Tp.index_worker();   // ID of the worker that executes the job
            vector_t x_curr;
            double obj_curr = std::numeric_limits<double>::max();
            vector_t x;
            double obj = std::numeric_limits<double>::max();
            int start = i * granularity;   // compute local start
            int end = 0;
            // last job could have granularity = granularity_last_job
            if (granularity_last_job != 0) {
                end = (i != (n_job - 1)) ? (i + 1) * granularity : start + granularity_last_job;
            } else {
                end = (i + 1) * granularity;
            }
            // search in sub-grid
            for (int j = start; j < end; j++) {
                grid_.row(j).assign_to(x_curr.transpose());
                obj_curr = objective(x_curr);
                // update minimum of worker if better optimum found
                if (obj_curr < obj) {
                    obj = obj_curr;
                    x = x_curr;
                }
            }
            // Optionally insert the worker's optimum into the Aligned vector at the end of the job
            if (obj < value_optimum_workers[index_worker].first) {
                value_optimum_workers[index_worker].first = obj;
                value_optimum_workers[index_worker].second = x;
            }
        });

        // reduce of value_optimum_workers[], min in value_ argmin in optimum_
        value_ = value_optimum_workers[0].first;
        optimum_ = value_optimum_workers[0].second;
        for (int i = 1; i < n_threads; i++) {
            if (value_optimum_workers[i].first < value_) {
                value_ = value_optimum_workers[i].first;
                optimum_ = value_optimum_workers[i].second;
            }
        }

        return optimum_;
    }

    // overload that create threadpool
    template <typename ObjectiveT, typename GridT>
        requires((internals::is_vector_like_v<GridT> || internals::is_matrix_like_v<GridT>))
    vector_t optimize(
      ObjectiveT&& objective, const GridT& grid, execution::execution_parallel,
      int n_threads = std::thread::hardware_concurrency(),
      int granularity =
        -1) {   // per ora int job_per_worker in input perche piu comodo fare i test poi sostituire valore scelto
        fdapde::threadpool Tp(1024, n_threads);
        return optimize(std::forward<ObjectiveT>(objective), grid, execution::par, Tp, granularity);
    }
    // observers
    const vector_t& optimum() const { return optimum_; }
    double value() const { return value_; }
    const std::vector<double>& values() const { return values_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_GRID_SEARCH_H__
