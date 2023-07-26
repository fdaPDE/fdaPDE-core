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

#ifndef __WOLFE_LINE_SEARCH_H__
#define __WOLFE_LINE_SEARCH_H__

#include "../../utils/symbols.h"

namespace fdapde {
namespace core {

// implementation of the Wolfe line search method for step selection
// check "Jorge Nocedal, Stephen J. Wright (2006), Numerical Optimization"
class WolfeLineSearch {
   private:
    double alpha_ = 1.0;
    double alpha_max_ = std::numeric_limits<double>::infinity(), alpha_min_ = 0;
    double c1_ = 1e-4, c2_ = 0.5;
   public:
    // constructors
    WolfeLineSearch() = default;
    WolfeLineSearch(double alpha, double c1, double c2) : alpha_(alpha), c1_(c1), c2_(c2) {};

    // bisection method for the weak Wolfe conditions
    template <typename Opt, typename Obj> bool pre_update_step(Opt& opt, Obj& obj) {
        // restore to initial value
        double alpha = alpha_;
        double alpha_max = alpha_max_, alpha_min = alpha_min_;
        double c1 = c1_, c2 = c2_;

	// initialization
        bool stop = false;
        double m = opt.grad_old.dot(opt.update);
        while (!stop) {
            if (obj(opt.x_old) - obj(opt.x_old + alpha * opt.update)   // Armijo–Goldstein condition
                + c1 * alpha * m < 0) {
                alpha_max = alpha;
                alpha = (alpha_min + alpha_max) * 0.5;
            } else if (obj.derive()(opt.x_old + alpha * opt.update).dot(opt.update) < c2 * m) {   // curvature condition
                alpha_min = alpha;
                alpha = (std::isinf(alpha_max)) ? 2 * alpha_min : (alpha_min + alpha_max) * 0.5;
            } else {
                stop = true;
            }
        }
        opt.h = alpha;
        return false;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __WOLFE_LINE_SEARCH_H__
