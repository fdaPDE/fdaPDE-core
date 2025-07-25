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

#ifndef __FDAPDE_NEWTON_H__
#define __FDAPDE_NEWTON_H__

#include "header_check.h"

namespace fdapde {

template <int N> class Newton {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_;
    double value_;                 // objective value at optimum
    int n_iter_ = 0;               // current iteration number
    std::vector<double> values_;   // explored objective values during optimization

    int max_iter_;   // maximum number of iterations before forced stop
    double tol_;     // tolerance on error before forced stop
    double step_;    // update step
    matrix_t hessian_;
    Eigen::PartialPivLU<matrix_t> inv_hessian_;
   public:
    static constexpr bool gradient_free = false;
    static constexpr int static_input_size = N;
    vector_t x_old, x_new, update, grad_old, grad_new;
    double h;
    // constructor
    Newton() : max_iter_(500), tol_(1e-5), step_(1e-2) { }
    Newton(int max_iter, double tol, double step) : max_iter_(max_iter), tol_(tol), step_(step) { }
    Newton(const Newton& other) : max_iter_(other.max_iter_), tol_(other.tol_), step_(other.step_) { }
    Newton& operator=(const Newton& other) {
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        step_ = other.step_;
        return *this;
    }
    template <typename ObjectiveT, typename... Callbacks>
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Callbacks&&... callbacks) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
        std::tuple<Callbacks...> callbacks_ {callbacks...};
        bool stop = false;   // asserted true in case of forced stop
        double error = std::numeric_limits<double>::max();
        int size = N == Dynamic ? x0.rows() : N;
        auto grad = objective.gradient();
        auto hess = objective.hessian();
        h = step_;
	n_iter_ = 0;
        x_old = x0, x_new = vector_t::Constant(size, NaN);
        grad_old = grad(x_old), grad_new = vector_t::Constant(size, NaN);
        hessian_ = hess(x_old);
	inv_hessian_.compute(hessian_);
        update = -inv_hessian_.solve(grad_old);
        stop |= internals::exec_grad_hooks(*this, objective, callbacks_);
        error = grad_old.norm();
        values_.push_back(objective(x_old));

        while (n_iter_ < max_iter_ && error > tol_ && !stop) {
            stop |= internals::exec_adapt_hooks(*this, objective, callbacks_);
            // update along descent direction
            x_new = x_old + h * update;
            grad_new = grad(x_new);
	    // prepare next iteration
	    hessian_ = hess(x_new);
            inv_hessian_.compute(hessian_);
            update = -inv_hessian_.solve(grad_new);
            stop |=
              (internals::exec_grad_hooks(*this, objective, callbacks_) || internals::exec_stop_if(*this, objective));
            x_old = x_new;
            grad_old = grad_new;
            error = grad_new.norm();
            values_.push_back(objective(x_old));
            n_iter_++;
        }
        optimum_ = x_old;
        value_ = objective(optimum_);
        return optimum_;
    }
    // getters
    const vector_t& optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
    const std::vector<double>& values() const { return values_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_NEWTON_H__
