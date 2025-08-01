#ifndef __FDAPDE_BARZILAI_BORWEIN__
#define __FDAPDE_BARZILAI_BORWEIN__

#include "../../../../../test/src/logger.h"
#include "header_check.h"
#include <limits>
#include <cmath>
#include <algorithm>  // for std::clamp

namespace fdapde {

template <int N, typename... Args>
class BarzilaiBorwein {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;

    std::tuple<Args...> callbacks_ {};
    vector_t optimum_;
    double value_;
    double step_;
    int max_iter_;
    double tol_;
    int n_iter_ = 0;
    double h_min_, h_max_;
    double h_lower_bound_ = 1e-30;
    double h_upper_bound_ = 1e3;
    const double max_update_norm_ = 1.0;

    const double fallback_step_ = 1e-2;

   public:
    vector_t x_old, x_new, update, grad_old, grad_new, s, y;
    double obj_old, obj_new;
    double h;

    // constructor
    BarzilaiBorwein() = default;
    BarzilaiBorwein(int max_iter, double tol, double step)
        requires(sizeof...(Args) != 0)
        : max_iter_(max_iter), tol_(tol), step_(step) { }
    BarzilaiBorwein(int max_iter, double tol, double step, Args&&... callbacks)
        : callbacks_(std::make_tuple(std::forward<Args>(callbacks)...)), max_iter_(max_iter), tol_(tol), step_(step) { }

    BarzilaiBorwein(const BarzilaiBorwein& other)
        : callbacks_(other.callbacks_), max_iter_(other.max_iter_), tol_(other.tol_), step_(other.step_) { }
    BarzilaiBorwein& operator=(const BarzilaiBorwein& other) {
        callbacks_ = other.callbacks_;
        max_iter_ = other.max_iter_;
        step_ = other.step_;
        tol_ = other.tol_;
        return *this;
    }

    template <typename ObjectiveT, typename... Functor>
    requires(sizeof...(Functor) < 2) && ((requires(Functor f, double opt_old, double opt_new, double h) { f(opt_old, opt_new, h); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Functor&&... func) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);

        bool stop = false;
        vector_t zero = vector_t::Zero(x0.rows());
        double error = std::numeric_limits<double>::max();
        n_iter_ = 0;

        auto grad = objective.derive();

        x_old = x_new = x0;
        grad_old = grad_new = grad(x_new);
        obj_old = obj_new = objective(x_new);

        if (grad_old.isApprox(zero)) {   // already at stationary point
            optimum_ = x_new;
            value_ = obj_new;
            if constexpr (sizeof...(Functor) == 1) { (func(obj_old, obj_new, h), ...); }
            return optimum_;
        }
        error = grad_old.norm();

        h = step_;  // Initial step
        h_min_ = h;
        h_max_ = h;

        int n_dofs = x_new.size() / 3;

        // std::cout << "h: " << std::scientific << std::setprecision(5) << h;

        // pre-update checks
        update = - grad_new;
        stop |= execute_pre_update_step(*this, objective, callbacks_);
        if (stop) { std::cout << "pre_update " << std::endl; }

        while (n_iter_ < max_iter_ && error > 1e-18 && !stop) {

            // set-up new iteration
            x_old = x_new;
            grad_old = grad_new;
            obj_old = obj_new;

            // update direction
            update = - h * grad_old;

            // Optional update norm cap
            if (update.norm()/n_dofs > max_update_norm_) {
                update *= (n_dofs * max_update_norm_ / update.norm());
            }

            // Optional pre-update step (e.g. backtracking line search)
            // stop = stop || execute_pre_update_step(*this, objective, callbacks_);
            // if (stop) std::cout << "pre update stopping criterion " << stop << std::endl;

            // update
            x_new = x_old + update;
            grad_new = grad(x_new);
            obj_new = objective(x_new);

            // inspection print
            if constexpr (sizeof...(Functor) == 1) { (func(obj_old, obj_new, h), ...); }

            // std::cout << "h: " << std::scientific << std::setprecision(5) << h;

            s = x_new - x_old;
            y = grad_new - grad_old;
            double sy = s.dot(y);
            double ss = s.squaredNorm();

            double h_temp;
            if (sy > 1e-25) {
                h_temp = ss / sy;  // BB1 step
            } else {
                update = - grad_new;
                stop |= execute_pre_update_step(*this, objective, callbacks_);
                // td::cout << " -> [BTLS]";
            }
            h = h_temp;

            h = std::clamp(h, h_lower_bound_, h_upper_bound_);

            error = grad_new.norm();
            if (error < 1e-18) { std::cout << " error" << std::endl; }
            stop |= (execute_post_update_step(*this, objective, callbacks_) || execute_stopping_criterion(*this, objective));
            // if (stop) std::cout << "post-update " << stop << std::endl;

            // Bookkeeping
            n_iter_++;
            h_min_ = std::min(h_min_, h);
            h_max_ = std::max(h_max_, h);
        }

        if (n_iter_ == max_iter_) { std::cout << "max_iter " << std::endl; }
        optimum_ = x_new;
        value_ = obj_new;
        // std::cout << "h: " << std::scientific << std::setprecision(5) << step_;
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

#endif // __FDAPDE_BARZILAI_BORWEIN__