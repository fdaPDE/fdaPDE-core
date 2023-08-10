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

#ifndef __SPLINE_BASIS_H__
#define __SPLINE_BASIS_H__

#include "../../utils/symbols.h"
#include "spline.h"

namespace fdapde {
namespace core {

// a spline basis of order R built over a given set of knots
template <int R> class SplineBasis {
   private:
    DVector<double> knots_ {};   // vector of knots
    std::vector<Spline<R>> basis_ {};
   public:
    using const_iterator = typename std::vector<Spline<R>>::const_iterator;
    static constexpr std::size_t order = R;
    typedef Spline<R> ElementType;

    // constructor
    SplineBasis() = default;
    SplineBasis(const DVector<double>& knots) : knots_(knots) {
        // reserve space
        std::size_t n = knots.size();
        knots_.resize(n + 2 * R);
        // pad the knot vector to obtain a full basis for the whole knot span [u_0, u_n]
        for (std::size_t i = 0; i < n + 2 * R; ++i) {
            if (i < R) {
                knots_[i] = knots[0];
            } else {
                if (i < n + R) {
                    knots_[i] = knots[i - R];
                } else {
                    knots_[i] = knots[n - 1];
                }
            }
	}
        // reserve space and compute spline basis
        basis_.reserve(knots_.rows() - R - 1);
        for (std::size_t k = 0; k < knots_.size() - R - 1; ++k) {
            basis_.emplace_back(knots_, k);   // create spline centered at k-th point of knots_
        }
    }

    // getters
    const Spline<R>& operator[](std::size_t i) const { return basis_[i]; }
    int size() const { return basis_.size(); }
    const DVector<double>& knots() const { return knots_; }

    // iterators
    const_iterator begin() const { return basis_.cbegin(); }
    const_iterator end() const { return basis_.cend(); }
};

}   // namespace core
}   // namespace fdapde

#endif   // __SPLINE_BASIS_H__
