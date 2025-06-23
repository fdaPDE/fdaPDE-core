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

#ifndef __FDAPDE_LBFGS_H__
#define __FDAPDE_LBFGS_H__

#include "header_check.h"

namespace fdapde {

// Implementation of Broyden–Fletcher–Goldfarb–Shanno algorithm for unconstrained nonlinear optimization with optimized
// memory usage
template <int N, typename... Args> class LBFGS {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    std::tuple<Args...> callbacks_;
    vector_t optimum_;
    double value_;           // objective value at optimum
    int max_iter_;           // maximum number of iterations before forced stop
    int n_iter_ = 0;         // current iteration number
    double tol_;             // tolerance on error before forced stop
    double step_;            // update step
    int mem_size_ = 10;   // number of vector used for approximating the objective hessian
    Eigen::Matrix<double, Dynamic, Dynamic> grad_mem_, x_mem_;
   public:
    vector_t x_old, x_new, update, grad_old, grad_new;
    double h;
    // constructors
    LBFGS() = default;
    LBFGS(int max_iter, double tol, double step, int mem_size)
    requires(sizeof...(Args) != 0) : max_iter_(max_iter), tol_(tol), step_(step), mem_size_(mem_size) {
        fdapde_assert(mem_size_ >= 0);
    }
    LBFGS(int max_iter, double tol, double step, int mem_size, Args&&... callbacks) :
        callbacks_(std::make_tuple(std::forward<Args>(callbacks)...)),
        max_iter_(max_iter),
        tol_(tol),
        step_(step),
        mem_size_(mem_size) {
        assert(mem_size_ >= 0);
    }
    // copy semantic
    LBFGS(const LBFGS& other) :
        callbacks_(other.callbacks_),
	max_iter_(other.max_iter_), tol_(other.tol_), step_(other.step_), mem_size_(other.mem_size_) { }
    LBFGS& operator=(const LBFGS& other) {
        callbacks_ = other.callbacks_;
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        step_ = other.step_;
        mem_size_ = other.mem_size_;
        return *this;
    }

    template <typename ObjectiveT, typename... Functor>
        requires(sizeof...(Functor) < 2) && ((requires(Functor f, double value) { f(value); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Functor&&... func) {
        fdapde_static_assert(
            std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
            INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);
        bool stop = false;   // asserted true in case of forced stop
        double error = 0;
        double gamma = 1.0;
        auto grad = objective.gradient();
        n_iter_ = 0;
        h = step_;
        x_old = x0, x_new = x0;
        vector_t zero;
        if constexpr (N == Dynamic) {
            zero = vector_t::Zero(x0.rows());
        } else {
            zero = vector_t::Zero();
        }
        update = zero;
        grad_old = grad(x_old);
        if (grad_old.isApprox(zero)) {   // already at stationary point
            optimum_ = x_old;
            value_ = objective(optimum_);
            if constexpr (sizeof...(Functor) == 1) { (func(value_), ...); }
            return optimum_;
        }
        error = grad_old.norm();
        x_mem_.resize(x0.rows(), mem_size_);
        grad_mem_.resize(x0.rows(), mem_size_);

        while (n_iter_ < max_iter_ && error > tol_ && !stop) {
            // compute update direction
            vector_t q = grad_old;
            int current_mem = n_iter_ < mem_size_ ? n_iter_ : mem_size_;
            std::vector<double> alpha(current_mem, 0);
            for (int i = 0; std::cmp_less(i, current_mem); ++i) {
                int k = (n_iter_ + mem_size_ - i - 1) % mem_size_;
                alpha[i] = x_mem_.col(k).dot(q) / grad_mem_.col(k).dot(x_mem_.col(k));
                q -= alpha[i] * grad_mem_.col(k);
            }
            // H_0^k = I (initial guess of the inverse hessian)
            update = -gamma * q;
            for (int i = current_mem - 1; i >= 0; --i) {
                int k = (n_iter_ + mem_size_ - i - 1) % mem_size_;
                double beta = grad_mem_.col(k).dot(update) / grad_mem_.col(k).dot(x_mem_.col(k));
                update -= x_mem_.col(k) * (alpha[i] + beta);
            }
            stop |= execute_pre_update_step(*this, objective, callbacks_);
            // update along descent direction
            x_new = x_old + h * update;
            grad_new = grad(x_new);
            if (grad_new.isApprox(zero)) {   // already at stationary point
                optimum_ = x_old;
                value_ = objective(optimum_);
                if constexpr (sizeof...(Functor) == 1) { (func(value_), ...); }
                return optimum_;
            }
            // mem update
            // update inverse Hessian approximation
            int col_idx = n_iter_ % mem_size_;
            grad_mem_.col(col_idx) = grad_new - grad_old;
            x_mem_.col(col_idx) = x_new - x_old;
            gamma = x_mem_.col(col_idx).dot(grad_mem_.col(col_idx)) / grad_mem_.col(col_idx).norm();
            // prepare next iteration
            if constexpr (sizeof...(Functor) == 1) { (func(objective(x_old)), ...); }
            error = grad_new.norm();
            stop |=
              (execute_post_update_step(*this, objective, callbacks_) || execute_stopping_criterion(*this, objective));
            x_old = x_new;
            grad_old = grad_new;
            ++n_iter_;
        }
        optimum_ = x_old;
        value_ = objective(optimum_);
        if constexpr (sizeof...(Functor) == 1) { (func(value_), ...); }
        return optimum_;
    }
    // observers
    vector_t optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_LBFGS_H__
