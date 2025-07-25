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


/**
 * @brief Implementation of Broyden–Fletcher–Goldfarb–Shanno algorithm for unconstrained nonlinear optimization
 * 
 * @tparam N dimension of the problem
 * @tparam Args list of callbacks that will be called at different points of the algorithm. 
 */
template <int N> class LBFGS {
private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_;
    double value_;               // objective value at optimum
    int n_iter_ = 0;             // current iteration number
    std::vector<double> values_; // explored objective values during optimization

    int max_iter_;         // maximum number of iterations before forced stop
    double tol_;           // tolerance on error before forced stop
    double step_;          // update step
    int mem_size_ = 10;    // number of vector used for the hessian approx

    Eigen::Matrix<double, Dynamic, Dynamic> delta_grad_memory_, delta_x_memory_;

public:
    static constexpr bool gradient_free = false;
    static constexpr int static_input_size = N;
    vector_t x_old, x_new, update, grad_old, grad_new;
    double h;

private:
    vector_t compute_hess_inv_grad_mul_(const vector_t &grad, double initial_coeff) {
        vector_t q = grad;

        // If the iteration count is too low, then we cannot find [mem_size_] vectors,
        // so we instead use the last [n_iter_] to compute the approximation of the hessian.
        int current_memory = n_iter_ < mem_size_ ? n_iter_: mem_size_;
        
        Eigen::VectorXd alphas (current_memory);
        alphas.setZero();
        
        for(int i = 0; std::cmp_less(i, current_memory); ++i) {
            int k = (n_iter_ + mem_size_ - i - 1) % mem_size_;
            vector_t delta_x = delta_x_memory_.col(k);
            vector_t delta_grad = delta_grad_memory_.col(k);

            alphas(i) = delta_x.dot(q) / delta_grad.dot(delta_x);
            q -= alphas(i) * delta_grad;
        }

        // H_0^k (Our initial guess as the inverse hessian at x_old) is chosen to be
        // an identity matrix times a constant anyways so we don't have to store any matrix
        vector_t r = initial_coeff /* \times Identity \times */ * q;

        for(int i = current_memory - 1; i >= 0; --i) {
            int k = (n_iter_ + mem_size_ - i - 1) % mem_size_;
            vector_t delta_x = delta_x_memory_.col(k);
            vector_t delta_grad = delta_grad_memory_.col(k);
            auto beta = delta_grad.dot(r) / delta_grad.dot(delta_x);
            r += delta_x*(alphas(i) - beta);
        }

        return r;
    }

public:
    // constructors
    LBFGS(): 
        max_iter_(500), tol_(1e-5), step_(1e-2), mem_size_(5) {
    }

    /**
     * @brief Construct a new LBFGS instance
     * 
     * @param max_iter maximum number of iterations performed
     * @param memory_size size of the memory used to approximate the inverse hessian matrix in the LBFGH alg.
     * @param tol tolerance used as the stoping criterion (|\nabla f_k|_2 < tol)
     * @param step default step used, may be overwritten by the line search method
     */
    LBFGS(int max_iter, double tol, double step, int memory_size) :
        max_iter_(max_iter),
        mem_size_(memory_size),
        tol_(tol), step_(step){
        fdapde_assert(mem_size_ >= 0);
    }

    // copy semantic
    LBFGS(const LBFGS& other) :
        mem_size_(other.mem_size_),
        max_iter_(other.max_iter_),
        n_iter_(other.n_iter_),
        tol_(other.tol_),
        delta_grad_memory_(other.delta_grad_memory_),
        delta_x_memory_(other.delta_x_memory_),
        value_(other.value_),
        step_(other.step_),
        optimum_(other.optimum_),
        values_(other.values_) {
    }
    
    LBFGS& operator=(const LBFGS& other) {
        mem_size_ = other.mem_size_;
        max_iter_ = other.max_iter_;
        n_iter_ = other.n_iter_;
        tol_ = other.tol_;
        delta_grad_memory_ = other.delta_grad_memory_;
        delta_x_memory_ = other.delta_x_memory_;
        value_ = other.value_;
        step_ = other.step_;
        optimum_ = other.optimum_;
        values_ = other.values_;
        return *this;
    }

    /**
     * @brief Minimizes a function using the L-BFGS method
     * 
     * @tparam ObjectiveT Type of the function to be minimized
     * @tparam Functor Types of the callbacks
     * @param objective function to be minimized
     * @param x0 initial point
     * @param func
     */
    template <typename ObjectiveT, typename... Callbacks>
    vector_t optimize(
        ObjectiveT&& objective,
        const vector_t& x0,
        Callbacks&&... callbacks
    ) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE);
            
        // whether or not the callbacks's stoping criteria are met
        bool stop = false;

        // Callback management
        std::tuple<Callbacks...> callbacks_ {callbacks...};
        
        // Initialize the state
        vector_t zero;
        if constexpr (N == Dynamic) {
            zero = vector_t::Zero(x0.rows());
        } else {
            zero = vector_t::Zero();
        }
        double error = 0;
        auto grad = objective.gradient();
        n_iter_ = 0;
        update = zero;
        h = step_;
        x_old = x0, x_new = x0;
        delta_x_memory_.resize(x0.rows(), mem_size_);
        delta_grad_memory_.resize(x0.rows(), mem_size_);

        // Stationarity test
        grad_old = grad(x_old);
        stop |= internals::exec_grad_hooks(*this, objective, callbacks_);
        if (grad_old.isApprox(zero)) {
            optimum_ = x_old;
            value_ = objective(optimum_);
            return optimum_;
        }
        error = grad_old.norm();
        
        // state to calculate the (inverse Hessian * grad) approx
        double initial_coeff = 1.0;
        
        while (n_iter_ < max_iter_ && error > tol_ && !stop) {
            // Update along descent direction
            update = -compute_hess_inv_grad_mul_(grad_old, initial_coeff);
            stop |= internals::exec_adapt_hooks(*this, objective, callbacks_);
            x_new = x_old + h * update;

            // Stationarity test
            grad_new = grad(x_new);
            if (grad_new.isApprox(zero)) {
                optimum_ = x_old;
                value_ = objective(optimum_);
                return optimum_;
            }

            // Update the memory
            auto x_diff = x_new - x_old;
            auto grad_diff = grad_new - grad_old;

            delta_grad_memory_.col(n_iter_ % mem_size_) = grad_diff;
            delta_x_memory_.col(n_iter_ % mem_size_) = x_diff;

            // Calculate the inverse Hessian approximation
            initial_coeff = x_diff.dot(grad_diff) / grad_diff.dot(grad_diff);
            
            // Prepare next iteration
            stop |= (internals::exec_grad_hooks(*this, objective, callbacks_) || internals::exec_stop_if(*this, objective));
            
            error = grad_new.norm();
            values_.push_back(objective(x_old));
            x_old = x_new;
            grad_old = grad_new;
            ++n_iter_;
        }	

        optimum_ = x_old;
        value_ = values_.back();
        return optimum_;
    }
    
    /**
     * @brief Returns the argmin of the last call to [optimize]
     * 
     * @return vector_t 
     */
    vector_t optimum() const { return optimum_; }

    /**
     * @brief Returns the minimum of the last call to [optimize]
     * 
     * @return double 
     */
    double value() const { return value_; }

    /**
     * @brief Returns the number of iterations performed by the method on the last call to [optimize]
     * 
     * @return int 
     */
    int n_iter() const { return n_iter_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_LBFGS_H__
