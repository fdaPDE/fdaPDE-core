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

#ifndef __NEWTON_H__
#define __NEWTON_H__

#include "../fields.h"
#include "../utils/symbols.h"
#include "callbacks/callbacks.h"

namespace fdapde {
namespace core {

// implementation of the newton method for unconstrained nonlinear optimization
template <unsigned int N> class Newton {
   private:
    std::size_t max_iter_;   // maximum number of iterations before forced stop
    double tol_;             // tolerance on error before forced stop
    double step_;            // update step

    SVector<N> optimum_;
    double value_;   // objective value at optimum
   public:
    SVector<N> x_old, x_new, update, grad_old, grad_new;
    SMatrix<N> hessian;
    Eigen::PartialPivLU<SMatrix<N>> inv_hessian;
    double h;

    // constructor
    Newton() = default;
    Newton(std::size_t max_iter, double tol, double step) : max_iter_(max_iter), tol_(tol), step_(step) {};

    template <typename F, typename... Args> void optimize(F& objective, const SVector<N>& x0, Args... args) {
        static_assert(
          std::is_same<decltype(std::declval<F>().operator()(SVector<N>())), double>::value,
          "cannot find definition for F.operator()(const SVector<N>&)");

        bool stop = false;   // asserted true in case of forced stop
        std::size_t n_iter = 0;
        double error = std::numeric_limits<double>::max();
        h = step_;   // restore optimizer step

        x_old = x0;
        x_new = x0;

        while (n_iter < max_iter_ && error > tol_ && !stop) {
            grad_old = objective.derive()(x_old);
            hessian = objective.derive_twice()(x_old);

            inv_hessian.compute(hessian);
            update = -inv_hessian.solve(grad_old);
            stop |= execute_pre_update_step(*this, objective, args...);

            // update along descent direction
            x_new = x_old + h * update;
            grad_new = objective.derive()(x_new);

            // prepare next iteration
            error = grad_new.norm();
            stop |= execute_post_update_step(*this, objective, args...);
            x_old = x_new;
            grad_old = grad_new;
            n_iter++;
        }
        optimum_ = x_old;
        value_ = objective(optimum_);
        return;
    }

    // getters
    SVector<N> optimum() const { return optimum_; }
    double value() const { return value_; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __NEWTON_H__
