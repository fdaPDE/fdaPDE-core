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

#ifndef __FDAPDE_BSPLINE_BASIS_H__
#define __FDAPDE_BSPLINE_BASIS_H__

#include "header_check.h"

namespace fdapde {

// given vector of knots u_1, u_2, ..., u_N, this class represents the set of N + order - 1 spline basis functions
// {l_1(x), l_2(x), ..., l_{N + order - 1}(x)} centered at knots u_1, u_2, ..., u_N
class BSplineBasis {
   private:
    int order_;
    std::vector<Spline> basis_ {};
    std::vector<double> knots_ {};
   public:
    static constexpr int StaticInputSize = 1;
    static constexpr int Order = Dynamic;
    // constructors
    constexpr BSplineBasis() : order_(0) { }
    // constructor from user defined knot vector
    template <typename KnotsVectorType>
        requires(requires(KnotsVectorType knots, int i) {
                    { knots[i] } -> std::convertible_to<double>;
                    { knots.size() } -> std::convertible_to<std::size_t>;
                })
    BSplineBasis(KnotsVectorType&& knots, int order) : order_(order) {
        fdapde_assert(
          std::is_sorted(knots.begin() FDAPDE_COMMA knots.end(), std::less_equal<double>()), std::invalid_argument,
          "spline knots must be strictly increasing");
        int n = knots.size();
        knots_.resize(n);
        for (int i = 0; i < n; ++i) { knots_[i] = knots[i]; }
        // define basis system
        basis_.reserve(n - order_ + 1);
        for (int i = 0; i < n - order_ - 1; ++i) { basis_.emplace_back(knots_, i, order_); }
    }
    // constructor from geometric interval (no repeated knots)
    BSplineBasis(const Triangulation<1, 1>& interval, int order) : order_(order) {
        // construct knots vector
        Eigen::Matrix<double, Dynamic, 1> knots = interval.nodes();
        fdapde_assert(
          std::is_sorted(knots.begin() FDAPDE_COMMA knots.end() FDAPDE_COMMA std::less_equal<double>()),
          std::invalid_argument, "spline knots must be strictly increasing");
        int n = knots.size();
        knots_.resize(n + 2 * order_);
        // pad the knot vector to obtain a full basis for the whole knot span [knots[0], knots[n-1]]
        for (int i = 0; i < n + 2 * order_; ++i) {
            if (i < order_) {
                knots_[i] = knots[0];
            } else {
                if (i < n + order_) {
                    knots_[i] = knots[i - order_];
                } else {
                    knots_[i] = knots[n - 1];
                }
            }
	}
        // define basis system
        basis_.reserve(knots_.size() - order_ + 1);
        for (std::size_t i = 0; i < knots_.size() - order_ - 1; ++i) { basis_.emplace_back(knots_, i, order_); }
    }
    // getters
    constexpr const Spline& operator[](int i) const { return basis_[i]; }
    constexpr int size() const { return basis_.size(); }
    constexpr const std::vector<double>& knots_vector() const { return knots_; }
    int n_knots() const { return knots_.size(); }
    int order() const { return order_; }
};

} // namespace fdapde

#endif // __FDAPDE_BSPLINE_BASIS_H__
