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

#ifndef __FDAPDE_GRADIENT_DESCENT__
#define __FDAPDE_GRADIENT_DESCENT__

#include "header_check.h"

namespace fdapde {

// implementation of Gradient Descent algorithm for unconstrained nonlinear optimization
template <int N, typename... Args> class GradientDescent {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    std::tuple<Args...> callbacks_ {};
    vector_t optimum_;
    double value_;     // objective value at optimum
    int max_iter_;     // maximum number of iterations before forced stop
    double tol_;       // tolerance on error before forced stop
    double step_;      // update step
    int n_iter_ = 0;   // current iteration number
   public:
    vector_t x_old, x_new, update, grad_old, grad_new;
    matrix_t inv_hessian;
    double h;
    // constructor
    GradientDescent() = default;
    GradientDescent(int max_iter, double tol, double step)
        requires(sizeof...(Args) != 0)
        : max_iter_(max_iter), tol_(tol), step_(step) { }
    GradientDescent(int max_iter, double tol, double step, Args&&... callbacks) :
        callbacks_(std::make_tuple(std::forward<Args>(callbacks)...)), max_iter_(max_iter), tol_(tol), step_(step) { }
    // copy semantic
    GradientDescent(const GradientDescent& other) :
        callbacks_(other.callbacks_), max_iter_(other.max_iter_), tol_(other.tol_), step_(other.step_) { }
    GradientDescent& operator=(const GradientDescent& other) {
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        step_ = other.step_;
        callbacks_ = other.callbacks_;
        return *this;
    }
    template <typename ObjectiveT, typename... Functor>
        requires(sizeof...(Functor) < 2) && ((requires(Functor f, double value) { f(value); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Functor&&... func) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);
        bool stop = false;   // asserted true in case of forced stop
        double error = std::numeric_limits<double>::max();
        h = step_;
        n_iter_ = 0;
        x_old = x0, x_new = x0;
	auto grad = objective.gradient();
        grad_old = grad(x_old);

        while (n_iter_ < max_iter_ && error > tol_ && !stop) {
            update = -grad_old;
            stop |= execute_pre_update_step(*this, objective, callbacks_);
            // update along descent direction
            x_new = x_old + h * update;
            grad_new = grad(x_new);
            if constexpr (sizeof...(Functor) == 1) { (func(objective(x_old)), ...); }
            // prepare next iteration
            error = grad_new.norm();
            stop |=
              (execute_post_update_step(*this, objective, callbacks_) || execute_stopping_criterion(*this, objective));
            x_old = x_new;
            grad_old = grad_new;
            n_iter_++;
        }
        optimum_ = x_old;
        value_ = objective(optimum_);
	if constexpr (sizeof...(Functor) == 1) { (func(value_), ...); }
        return optimum_;
    }
    // getters
    vector_t optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_GRADIENT_DESCENT__
