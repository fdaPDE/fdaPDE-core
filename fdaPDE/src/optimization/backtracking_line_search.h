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

#ifndef __FDAPDE_BACKTRACKING_LINE_SEARCH_H__
#define __FDAPDE_BACKTRACKING_LINE_SEARCH_H__

#include "header_check.h"

namespace fdapde {

// implementation of the backatracking line search method for step selection
class BacktrackingLineSearch {
private:
    double alpha_old_ = 1.0;     // last used step size
    double alpha_ = 1.0;     // last used step size
    double beta_ = 1/std::sqrt(2);      // shrink factor
    double gamma_ = 1e-6;    // Armijo constant
    double alpha_min_ = 1e-30;
    double alpha_max_ = 1e3;
    bool first_ = true;

public:

    template <typename Opt, typename Obj>
    bool pre_update_step(Opt& opt, Obj& obj) {

        double alpha;
        if (first_) {
            first_ = false;
            alpha = opt.step();
        } else alpha = alpha_;

        // std::cout << "h: " << std::scientific << std::setprecision(5) << alpha << " -> ";

        double m = opt.grad_old.dot(opt.update);
        if (m >= 0) {
            // In some methods, such as Adams, we should allow to move in a non-optimal direction.
            // We could add a flag in each optimizer that says if moving in non-optimal directions is allowed or not
            // opt.h = 0.001; // default Adam
            // std::cout << ",\tthis is not a descent direction\n";
            // return true; // not a descent direction
            return false;
        }

        double f_old = opt.obj_old;
        bool reduced = false;

        int n_dofs = opt.x_old.size() / 3;
        double d_norm = opt.update.norm() / n_dofs;
        alpha = 1 / d_norm;
        while (obj(opt.x_old + alpha * opt.update) > f_old + gamma_ * alpha * m && alpha > alpha_min_) {
            alpha *= beta_;
            reduced = true;
        }

        // Set the step size (opt.h) to alpha, but not smaller than alpha_min_
        opt.h = alpha < alpha_min_ ? alpha_min_ : alpha;

        // If the step was reduced and alpha_ hasn't changed from alpha_old_,
        // but the change from alpha is significant,
        // update alpha_ to alpha / beta_ (the step before the one currently selected), otherwise leave alpha_ unchanged
        alpha_ = (reduced && std::abs(alpha_ - alpha_old_) < alpha_min_ &&
                  std::abs(alpha_ * beta_ - alpha) > alpha_min_) ? alpha / beta_ : alpha_;

        // Store the current alpha_ value as alpha_old_ for the next iteration
        alpha_old_ = alpha_;

        // If the step was not reduced, potentially increase alpha_,
        // but cap it to alpha_max_. The increase factor is beta_^2.
        alpha_ = reduced ? alpha_ : std::min(alpha / std::pow(beta_, 2), alpha_max_);
        return false;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_BACKTRACKING_LINE_SEARCH_H__
