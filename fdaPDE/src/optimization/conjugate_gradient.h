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

#ifndef __FDAPDE_CONJUGATE_GRADIENT_H__
#define __FDAPDE_CONJUGATE_GRADIENT_H__

#include "header_check.h"
#include <limits>
#include <tuple>
#include <utility>
#include <cmath>
#include <algorithm>

namespace fdapde {

// implementation of Conjugate Gradient method for unconstrained nonlinear optimization
template <int N, typename... Args>
class ConjugateGradient {
   private:
    using vector_t =
        std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
        std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    std::tuple<Args...> callbacks_ {};
    vector_t optimum_;
    double value_;     // objective value at optimum
    int max_iter_;     // maximum number of iterations before forced stop
    double tol_;       // tolerance on error before forced stop
    double step_;      // initial step size
    double h_min_;
    double h_max_;
    int n_iter_ = 0;   // current iteration number
    bool use_polak_ribiere_ = true;

   public:
    vector_t x_old, x_new, update, dir, grad_old, grad_new;
    double obj_old, obj_new;
    double h;

    // constructors
    ConjugateGradient() = default;

    ConjugateGradient(int max_iter, double tol, double step, bool use_polak_ribiere = true)
        requires(sizeof...(Args) != 0)
        : max_iter_(max_iter), tol_(tol), step_(step), use_polak_ribiere_(use_polak_ribiere) { }

    ConjugateGradient(int max_iter, double tol, double step, bool use_polak_ribiere, Args&&... callbacks)
        : callbacks_(std::make_tuple(std::forward<Args>(callbacks)...)),
          max_iter_(max_iter),
          tol_(tol),
          step_(step),
          use_polak_ribiere_(use_polak_ribiere) { }

    // copy semantic
    ConjugateGradient(const ConjugateGradient& other)
        : callbacks_(other.callbacks_),
          max_iter_(other.max_iter_),
          tol_(other.tol_),
          step_(other.step_),
          use_polak_ribiere_(other.use_polak_ribiere_) { }

    ConjugateGradient& operator=(const ConjugateGradient& other) {
        if (this != &other) {
            callbacks_ = other.callbacks_;
            max_iter_ = other.max_iter_;
            tol_ = other.tol_;
            step_ = other.step_;
            use_polak_ribiere_ = other.use_polak_ribiere_;
        }
        return *this;
    }

    // optimize
    template <typename ObjectiveT, typename... Functor>
    requires(sizeof...(Functor) < 2) && ((requires(Functor f, double opt_old, double opt_new, double h) { f(opt_old, opt_new, h); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Functor&&... func) {
        fdapde_static_assert(
            std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
            INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE
        );

        const double epsilon = 1e-12;
        const int restart_frequency = 10;
        const double max_objective = 1e10;
        const double max_step = step_;

        bool stop = false;
        double error = std::numeric_limits<double>::max();
        double beta = 0.0;
        h = step_;
        n_iter_ = 0;
        h_min_ = std::numeric_limits<double>::infinity();
        h_max_ = 0.0;

        x_old = x_new = x0;

        auto grad = objective.derive();
        grad_old = grad_new = grad(x_new);
        obj_old = obj_new = objective(x_new);
        dir = -grad_old;

        while (n_iter_ < max_iter_ && error > 1e-20 && !stop) {

            // Save history
            x_old = x_new;
            grad_old = grad_new;
            obj_old = obj_new;

            update = dir;

            // If direction is not descent, reset
            if (grad_old.dot(update) >= 0) {
                dir = -grad_old;
                update = dir;
                beta = 0.0;
            }

            stop = stop || execute_pre_update_step(*this, objective, callbacks_);

            // Perform update
            x_new = x_old + h * update;
            grad_new = grad(x_new);
            obj_new = objective(x_new);

            // Optional user monitor
            if constexpr (sizeof...(Functor) == 1) { (func(obj_old, obj_new, h), ...); }

            // Compute beta
            double denom = grad_old.dot(grad_old) + epsilon;
            if (use_polak_ribiere_) {
                beta = grad_new.dot(grad_new - grad_old) / denom;
            } else {
                beta = grad_new.dot(grad_new) / denom;
            }
            beta = std::max(0.0, beta); // enforce descent

            // Restart direction periodically
            if (n_iter_ % restart_frequency == 0) {
                dir = -grad_new;
            } else {
                dir = -grad_new + beta * dir;
            }

            // Prepare next iteration
            error = grad_new.norm();
            // if (error < 1e-20) { std::cout << "error " << std::endl; }

            stop = stop || execute_post_update_step(*this, objective, callbacks_) || execute_stopping_criterion(*this, objective);
            // if (stop) { std::cout << "post_update " << std::endl; }

            // bookkeeping
            n_iter_++;
            h_min_ = std::min(h_min_, h);
            h_max_ = std::max(h_max_, h);
        }

        // if (n_iter_ == max_iter_) { std::cout << "max_iter " << std::endl; }
        optimum_ = x_new;
        value_ = obj_new;
        // std::cout << "h: " << std::fixed << std::setprecision(10) << step_ << " -> ";
        if constexpr (sizeof...(Functor) == 1) { (func(obj_old, obj_new, h), ...); }
        return optimum_;
    }

    // getters
    vector_t optimum() const { return optimum_; }
    double step() const { return step_; }
    double h_min() const { return h_min_; }
    double h_max() const { return h_max_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
    // setter
    void set_step(double step) { step_ = step; }
    void set_max_iter(int max_iter) { max_iter_ = max_iter; }
    void reset_n_iter() { n_iter_ = 0; }
};

} // namespace fdapde

#endif // __FDAPDE_CONJUGATE_GRADIENT_H__