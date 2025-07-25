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
    using grid_t = MdMap<const double, MdExtents<Dynamic, Dynamic>>;

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
        grid_.row(0).assign_to(x_curr);
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
            grid_.row(i).assign_to(x_curr);
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
    // observers
    const vector_t& optimum() const { return optimum_; }
    double value() const { return value_; }
    const std::vector<double>& values() const { return values_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_GRID_SEARCH_H__
