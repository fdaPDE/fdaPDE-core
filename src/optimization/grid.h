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

#ifndef __GRID_H__
#define __GRID_H__

#include "../fields.h"
#include "../utils/symbols.h"
#include "callbacks/callbacks.h"

namespace fdapde {
namespace core {

// searches for the point in a given grid minimizing a given nonlinear objective
template <int N> class Grid {
   private:
    SVector<N> optimum_;
    double value_;   // objective value at optimum
   public:
    SVector<N> x_current;

    // constructor
    Grid() = default;

    template <typename F, typename... Args>
    void optimize(F& objective, const std::vector<SVector<N>>& grid, Args&... args) {
        // can be set true by some extension to cause a foced stop at any point of the execution
        bool stop = false;
        // algorithm initialization
        x_current = grid[0];
        value_ = objective(x_current);
        optimum_ = x_current;
        // optimize field over supplied grid
        for (std::size_t i = 1; i < grid.size() && !stop; ++i) {
            x_current = grid[i];
            double x = objective(x_current);
            stop |= execute_post_update_step(*this, objective, args...);

            // update minimum if better optimum found
            if (x < value_) {
                value_ = x;
                optimum_ = x_current;
            }
        }
        return;
    }

    // getters
    SVector<N> optimum() const { return optimum_; }
    double value() const { return value_; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __GRID_H__
