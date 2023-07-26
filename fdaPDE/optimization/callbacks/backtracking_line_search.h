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

#ifndef __BACKTRACKING_LINE_SEARCH_H__
#define __BACKTRACKING_LINE_SEARCH_H__

#include "../../utils/symbols.h"

namespace fdapde {
namespace core {

// implementation of the backatracking line search method for step selection
class BacktrackingLineSearch {
   private:
    double alpha_ = 2.0;
    double beta_  = 0.5;
    double gamma_ = 0.5;
   public:
    // constructors
    BacktrackingLineSearch() = default;
    BacktrackingLineSearch(double alpha, double beta, double gamma) : alpha_(alpha), beta_(beta), gamma_(gamma) {};

    // backtracking based step search
    template <typename Opt, typename Obj> bool pre_update_step(Opt& opt, Obj& obj) {
        double alpha = alpha_;   // restore to user defined settings
        double m = opt.grad_old.dot(opt.update);
        if (m < 0) {                                                      // descent direction
            while (obj(opt.x_old) - obj(opt.x_old + alpha * opt.update)   // Armijo–Goldstein condition
		   + gamma_ * alpha * m < 0) {
                alpha *= beta_;
            }
        }
        opt.h = alpha;
        return false;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __BACKTRACKING_LINE_SEARCH_H__
