#ifndef __FDAPDE_LBFGS_H__
#define __FDAPDE_LBFGS_H__

#include "../../../../../test/src/logger.h"
#include "header_check.h"
#include <deque>

namespace fdapde {

template <int N, typename... Args>
class LBFGS {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;

    std::tuple<Args...> callbacks_;
    std::deque<vector_t> s_list_;  // x diffs
    std::deque<vector_t> y_list_;  // grad diffs
    std::deque<double> rho_list_;  // 1/(y^T s)

    vector_t optimum_;
    double value_;
    int max_iter_;
    int m_ = 10;  // memory size
    double tol_;
    double step_;
    double h_min_;
    double h_max_;
    int n_iter_ = 0;

   public:
    vector_t x_old, x_new, update, grad_old, grad_new;
    double obj_old, obj_new;
    double h;

    // constructors
    LBFGS() = default;
    LBFGS(int max_iter, double tol, double step, int memory = 10)
        requires(sizeof...(Args) != 0)
        : max_iter_(max_iter), tol_(tol), step_(step), m_(memory) { }

    LBFGS(int max_iter, double tol, double step, int memory, Args&&... callbacks)
        : callbacks_(std::make_tuple(std::forward<Args>(callbacks)...)),
          max_iter_(max_iter), tol_(tol), step_(step), m_(memory) { }

    // copy
    LBFGS(const LBFGS& other)
        : callbacks_(other.callbacks_),
          max_iter_(other.max_iter_),
          tol_(other.tol_),
          step_(other.step_),
          m_(other.m_) { }

    LBFGS& operator=(const LBFGS& other) {
        callbacks_ = other.callbacks_;
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        step_ = other.step_;
        m_ = other.m_;
        return *this;
    }

    template <typename ObjectiveT, typename... Functor>
    requires(sizeof...(Functor) < 2) && ((requires(Functor f, double opt_old, double opt_new, double h) { f(opt_old, opt_new, h); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Functor&&... func) {
        fdapde_static_assert(
            std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
            INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);

        h = step_;
        n_iter_ = 0;
        h_min_ = step_;
        h_max_ = 0.0;


        auto grad = objective.derive();

        x_old = x_new = x0;
        grad_old = grad_new = grad(x_new);
        obj_old = obj_new = objective(x_new);

        double error = std::numeric_limits<double>::max();
        bool stop = false;   // asserted true in case of forced stop


        // int n_dofs = x_new.size() / 3;
        // file << x_new[0] << ", " << x_new[n_dofs] << ", " << x_new[2 * n_dofs] << ", " << obj_new << "\n";


        while (n_iter_ < max_iter_ && error > 1e-20 && !stop) {

            // set-up new iteration
            grad_old = grad_new;
            x_old = x_new;
            obj_old = obj_new;

            // Two-loop recursion
            vector_t q = grad_old;
            std::vector<double> alpha;
            for (int i = static_cast<int>(s_list_.size()) - 1; i >= 0; --i) {
                double a = rho_list_[i] * s_list_[i].dot(q);
                alpha.push_back(a);
                q = q - a * y_list_[i];
            }

            double scaling = 1.0;
            if (!y_list_.empty()) {
                int last = static_cast<int>(y_list_.size()) - 1;
                scaling = s_list_[last].dot(y_list_[last]) / y_list_[last].dot(y_list_[last]);
            }

            vector_t r = scaling * q;

            for (size_t i = 0; i < s_list_.size(); ++i) {
                double b = rho_list_[i] * y_list_[i].dot(r);
                r = r + s_list_[i] * (alpha[s_list_.size() - 1 - i] - b);
            }

            update = -r;

            // Optional pre-update step (e.g. backtracking line search)
            stop = stop || execute_pre_update_step(*this, objective, callbacks_);
            // if (stop) std::cout << "pre update stopping criterion " << stop << std::endl;

            // update
            x_new = x_old + h * update;
            grad_new = grad(x_new);
            obj_new = objective(x_new);

            // Save history
            vector_t s = x_new - x_old;
            vector_t y = grad_new - grad_old;
            double yts = y.dot(s);

            if (yts > 1e-20) {
                if (s_list_.size() == static_cast<size_t>(m_)) {
                    s_list_.pop_front();
                    y_list_.pop_front();
                    rho_list_.pop_front();
                }
                s_list_.push_back(s);
                y_list_.push_back(y);
                rho_list_.push_back(1.0 / yts);
            }

            // Post-update
            if constexpr (sizeof...(Functor) == 1) { (func(obj_old, obj_new, h), ...); }

            error = grad_new.norm();
            if (error < 1e-20) { std::cout << " error" << tol_ << std::endl; }
            stop |= (execute_post_update_step(*this, objective, callbacks_) || execute_stopping_criterion(*this, objective));
            // if (stop) std::cout << "post-update " << stop << std::endl;

            // Bookkeeping
            n_iter_++;
            h_min_ = std::min(h_min_, h);
            h_max_ = std::max(h_max_, h);

            // file << x_new[0] << ", " << x_new[n_dofs] << ", " << x_new[2 * n_dofs] << ", " << obj_new << "\n";

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

}  // namespace fdapde

#endif  // __FDAPDE_LBFGS_H__