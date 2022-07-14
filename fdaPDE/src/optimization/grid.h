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

#ifndef __FDAPDE_GRID_H__
#define __FDAPDE_GRID_H__

#include "header_check.h"

namespace fdapde {

// searches for the point in a given grid minimizing a given nonlinear objective
template <int N, typename... Args> class GridOptimizer {
   private:
    using vector_t = std::conditional_t<N == Dynamic, std::vector<double>, std::array<double, N>>;
    using grid_t   = MdMap<const double, MdExtents<Dynamic, Dynamic>>;
  
    std::tuple<Args...> callbacks_ {};
    vector_t optimum_;
    double value_;   // objective value at optimum
    int size_;
    std::vector<double> values_;   // explored objective values during optimization
   public:
    vector_t x_current;
    // constructor
    GridOptimizer() requires(N != Dynamic && sizeof...(Args) != 0) : size_(N) { }
    GridOptimizer(int size) requires(N == Dynamic && sizeof...(Args) != 0) : size_(size) { }
    GridOptimizer(Args&&... callbacks)
        requires(N != Dynamic)
        : callbacks_(std::make_tuple(std::forward<Args>(callbacks)...)), size_(N) { }
    GridOptimizer(int size, Args&&... callbacks)
        requires(N == Dynamic)
        : callbacks_(std::make_tuple(std::forward<Args>(callbacks)...)), size_(size) { }
    // copy semantic
    GridOptimizer(const GridOptimizer& other) : callbacks_(other.callbacks_), size_(other.size_) { }
    GridOptimizer& operator=(const GridOptimizer& other) {
        callbacks_ = other.callbacks_;
	size_ = other.size_;
        return *this;
    }
    template <typename ObjectiveT, typename GridT, typename... Functor>
        requires((internals::is_vector_like_v<GridT> || internals::is_matrix_like_v<GridT>) &&
                 sizeof...(Functor) < 2) &&
                ((requires(Functor f, double value) { f(value); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const GridT& grid, Functor&&... func) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);
        grid_t grid_;
        if constexpr (internals::is_vector_like_v<GridT>) {
            fdapde_assert(grid.size() % size_ == 0);
            grid_ = grid_t(grid.data(), grid.size() / size_, size_);
        } else {
            fdapde_assert(grid.cols() == size_);
            grid_ = grid_t(grid.data(), grid.rows(), size_);
        }
        bool stop = false;   // asserted true in case of forced stop
        // algorithm initialization
        grid_.row(0).assign_to(x_current);
        value_ = objective(x_current);
        if constexpr (sizeof...(Functor) == 1) { (func(value_), ...); }
        values_.push_back(value_);
        optimum_ = x_current;
        // optimize field over supplied grid
        for (int i = 1; i < grid_.rows() && !stop; ++i) {
            grid_.row(i).assign_to(x_current);
            double x = objective(x_current);
            values_.push_back(x);
            stop |= execute_post_update_step(*this, objective, callbacks_);
            // update minimum if better optimum found
            if (x < value_) {
                value_ = x;
                optimum_ = x_current;
            }
            if constexpr (sizeof...(Functor) == 1) { (func(value_), ...); }
        }
        return optimum_;
    }
    // getters
    vector_t optimum() const { return optimum_; }
    double value() const { return value_; }
    const std::vector<double>& values() const { return values_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_GRID_H__
