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

template <typename VectorType_>
    requires(!std::is_lvalue_reference_v<VectorType_>)
struct GridSearch {
    using VectorType = VectorType_;
    static constexpr bool is_gradient_free = true;
    // publicly accessible optimizer state
    VectorType x_curr;
    double obj_curr;  
    // constructor
    GridSearch() : x_curr(), obj_curr(), optimum_(), value_(), values_() { }

    // optimizes objective over grid. sequential execution overload. for an n-dimensional grid of k points, grid is
    // organized as [(p_{i,1}, p_{i,2}, ..., p_{i,n})]_{i=1}^k
    template <typename ObjectiveType, typename GridType, typename... Callbacks>
        requires(
          std::is_invocable_v<ObjectiveType, VectorType> && fdapde::is_matrix_v<GridType> &&
          requires(GridType grid, int i) {
              { grid.row(i) } -> std::convertible_to<VectorType>;
          } && (is_opt_callback_v<Callbacks> && ...))
    VectorType
    optimize(fdapde::execution_seq_t, ObjectiveType&& objective, const GridType& grid, Callbacks&&... callbacks) {
        std::tuple<Callbacks...> callbacks_ {callbacks...};
        bool stop = false;   // asserted true in case of forced stop
        values_.reserve(grid.rows());

        x_curr = grid.row(0);
        obj_curr = objective(x_curr);
        stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
        values_.push_back(obj_curr);
        // initialize optimum
        value_ = obj_curr;
        optimum_ = x_curr;
        // optimize objective over grid
        for (int i = 1; i < grid.rows() && !stop; ++i) {
            x_curr = grid.row(i);
            obj_curr = objective(x_curr);
            stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
            values_.push_back(obj_curr);
            // optimum update
            if (obj_curr < value_) {
                value_ = obj_curr;
                optimum_ = x_curr;
            }
            stop |= internals::exec_stop_if(*this, objective);
        }
        return optimum_;
    }
    // default to sequential execution
    template <typename ObjectiveType, typename GridType, typename... Callbacks>
        requires(is_opt_callback_v<Callbacks> && ...)
    VectorType optimize(ObjectiveType&& objective, const GridType& grid, Callbacks&&... callbacks) {
        return optimize(
          fdapde::execution_seq, std::forward<ObjectiveType>(objective), grid, std::forward<Callbacks>(callbacks)...);
    }

    // optimizes objective over grid. parallel execution overload. for an n-dimensional grid of k points, grid is
    // organized as [(p_{i,1}, p_{i,2}, ..., p_{i,n})]_{i=1}^k
    template <typename ObjectiveType, typename GridType, typename... Callbacks>
        requires(
          std::is_invocable_v<ObjectiveType, VectorType> && fdapde::is_matrix_v<GridType> &&
          requires(GridType grid, int i) {
              { grid.row(i) } -> std::convertible_to<VectorType>;
          } && (is_opt_callback_v<Callbacks> && ...))
    VectorType
    optimize(fdapde::execution_par_t, ObjectiveType&& objective, const GridType& grid, Callbacks&&... callbacks) {
        std::tuple<Callbacks...> callbacks_ {callbacks...};
        bool stop = false;   // asserted true in case of forced stop
        values_.resize(grid.rows());
        const int n_threads = num_threads();

        // local worker state
        std::vector<double> local_value(n_threads, std::numeric_limits<double>::max());
        std::vector<int> local_optimum(n_threads);
        // parallel optimize objective over grid
        auto local_optimize = [&](int i) {
            const int w_id = this_worker_id();
            VectorType local_x_curr = grid.row(i);
            double local_obj_curr = objective(local_x_curr);
            stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
            values_[i] = local_obj_curr;
            // update minimum if better optimum found
            if (local_obj_curr < local_value[w_id]) {
                local_value[w_id] = local_obj_curr;
                local_optimum[w_id] = i;
            }
            stop |= internals::exec_stop_if(*this, objective);
        };
        parallel_for(0, grid.rows(), local_optimize);

        // sequential minimum reduction
        int j = 0;
        value_ = local_value[j];
        for (int i = 1; i < n_threads; ++i) {
            if (local_value[i] < value_) {
                value_ = local_value[i];
                j = i;
            }
        }
        optimum_ = grid.row(local_optimum[j]);
        return optimum_;
    }

    // observers
    const VectorType& optimum() const { return optimum_; }
    double value() const { return value_; }
    const std::vector<double>& values() const { return values_; }
   private:
    VectorType optimum_;   // optimum point
    double value_;         // objective value at optimum
    std::vector<double> values_;
    std::mutex m_;
};

}   // namespace fdapde

#endif   // __FDAPDE_GRID_SEARCH_H__
