#ifndef __FDAPDE_ADAM_OPTIMIZER__
#define __FDAPDE_ADAM_OPTIMIZER__

#include "../../../../../test/src/logger.h"
#include "header_check.h"
#include <cmath>

namespace fdapde {

// implementation of Adam algorithm for unconstrained nonlinear optimization
template <int N, typename... Args>
class Adam {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;

    std::tuple<Args...> callbacks_ {};
    vector_t optimum_;
    double value_;
    int max_iter_;
    double tol_;
    double step_;
    int n_iter_ = 0;
    double h_min_, h_max_;

    // Adam hyperparameters
    double beta1_ = 0.9;
    double beta2_ = 0.999;
    double epsilon_ = 1e-8;
    vector_t m_, v_;

   public:
    vector_t x_old, x_new, update, grad_old, grad_new;
    double obj_old, obj_new;
    double h;

    // constructor
    Adam() = default;
    Adam(int max_iter, double tol, double step)
        requires(sizeof...(Args) != 0)
        : max_iter_(max_iter), tol_(tol), step_(step) { }
    Adam(int max_iter, double tol, double step, Args&&... callbacks)
        : callbacks_(std::make_tuple(std::forward<Args>(callbacks)...)), max_iter_(max_iter), tol_(tol), step_(step) { }

    Adam(const Adam& other) :
        callbacks_(other.callbacks_), max_iter_(other.max_iter_), tol_(other.tol_), step_(other.step_) { }
    Adam& operator=(const Adam& other) {
        callbacks_ = other.callbacks_;
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        step_ = other.step_;
        return *this;
    }

    template <typename ObjectiveT, typename... Functor>
    requires(sizeof...(Functor) < 2) && ((requires(Functor f, double opt_old, double opt_new, double h) { f(opt_old, opt_new, h); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, // const vector_t& m0, const vector_t& v0,
                  Functor&&... func) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);

        bool stop = false;
        double error = std::numeric_limits<double>::max();
        h = step_;

        n_iter_ = 0;
        x_old = x0, x_new = x0;
        auto grad = objective.derive();
        grad_old = grad_new = grad(x_new);
        obj_old = obj_new = objective(x_new);
        error = grad_old.norm();

        m_ = vector_t::Zero(x0.size());
        v_ = vector_t::Zero(x0.size());

        h_min_ = step_;
        h_max_ = 0.0;

        while (n_iter_ < max_iter_ && error > 1e-20 && !stop) {

            // set-up new iteration
            x_old = x_new;
            obj_old = obj_new;
            grad_old = grad_new;

            // first and second moment estimate
            m_ = beta1_ * m_ + (1 - beta1_) * grad_old;
            v_ = beta2_ * v_ + (1 - beta2_) * grad_old.array().square().matrix();

            // bias correction
            vector_t m_hat = m_ / (1 - std::pow(beta1_, n_iter_ + 1));
            vector_t v_hat = v_ / (1 - std::pow(beta2_, n_iter_ + 1));

            // RMSprop update rule
            update = - m_hat.array() / (v_hat.array() + epsilon_).sqrt().array();

            // pre-update checks
            stop |= execute_pre_update_step(*this, objective, callbacks_);
            if (stop) { std::cout << "pre_update " << std::endl; }

            // update along descent direction
            x_new = x_old + h * update;
            grad_new = grad(x_new);
            obj_new = objective(x_new);

            // inspection print
            if constexpr (sizeof...(Functor) == 1) { (func(obj_old, obj_new, h), ...); }

            // post-update checks
            error = grad_new.norm();
            if (error < 1e-20) { std::cout << "error " << std::endl; }
            stop |= (execute_post_update_step(*this, objective, callbacks_) || execute_stopping_criterion(*this, objective));
            // if (stop) { std::cout << "post_update " << std::endl; }

            // bookkeeping
            n_iter_++;
            h_min_ = std::min(h_min_, h);
            h_max_ = std::max(h_max_, h);
        }

        if (n_iter_ == max_iter_) { std::cout << "max_iter " << std::endl; }
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

#endif // __FDAPDE_ADAM_OPTIMIZER__