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

#ifndef __FDAPDE_SPLINE_H__
#define __FDAPDE_SPLINE_H__

#include "header_check.h"

namespace fdapde {

class Spline : public ScalarFieldBase<1, Spline> {
   public:
    using Base = ScalarFieldBase<1, Spline>;
    static constexpr int StaticInputSize = 1;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0;
    static constexpr int Order = Dynamic;
    using Scalar = double;
    using InputType = Vector<Scalar, StaticInputSize>;
   private:
    // order k derivative functor
    class Derivative : public ScalarFieldBase<1, Derivative> {
       private:
        std::vector<double> knots_ {};
        int i_ = 0;
        int order_ = 0;
        int n_ = 1;
       public:
        using Base = ScalarFieldBase<1, Derivative>;
        using Scalar = double;
        using InputType = Vector<Scalar, StaticInputSize>;

        constexpr Derivative() = default;
        constexpr Derivative(const std::vector<double>& knots, int i, int order, int n) :
            knots_(knots), i_(i), order_(order), n_(n) { }

        // non-recursive implementation of order k-th spline derivative evaluation at point, as detailed in "Piegl, L.,
        // & Tiller, W. (2012). The NURBS book. Springer Science & Business Media. Algorithm A2.5 pag 77"
        template <typename InputType_>
            requires(internals::is_subscriptable<InputType_, int> || std::is_floating_point_v<InputType_>)
        constexpr Scalar operator()(const InputType_& p_) const {
            if (n_ > order_) { return 0.0; }   // derivative order higher than spline order
            double p;
            if constexpr (internals::is_subscriptable<InputType_, int>) {
                p = p_[0];
            } else {
                p = p_;
            }
            // local property: return 0 if p is outside the range of this basis function
            if (p < knots_[i_] || p >= knots_[i_ + order_ + 1]) return 0.0;
            // initialize triangular table for basis functions
            MdArray<double, MdExtents<Dynamic, Dynamic>> N(order_ + 1, order_ + 1);
            // initialize zeroth-degree basis functions
            for (int j = 0; j <= order_; ++j) {
                if (p >= knots_[i_ + j] && p < knots_[i_ + j + 1]) { N(j, 0) = 1.0; }
            }
            // triangular table computation
            for (int k = 1; k <= order_; k++) {
                double saved = 0.0;
                if (N(0, k - 1) != 0.0) { saved = (p - knots_[i_]) * N(0, k - 1) / (knots_[i_ + k] - knots_[i_]); }

                for (int j = 0; j < order_ - k + 1; ++j) {
                    double Uleft  = knots_[i_ + j + 1];
                    double Uright = knots_[i_ + j + k + 1];
                    if (N(j + 1, k - 1) == 0.0) {
                        N(j, k) = saved;
                        saved = 0.0;
                    } else {
                        double temp = N(j + 1, k - 1) / (Uright - Uleft);
                        N(j, k) = saved + (Uright - p) * temp;
                        saved = (p - Uleft) * temp;
                    }
                }
            }
            // nth derivative computation
            std::vector<double> ND(n_ + 1);
            for (int j = 0; j <= n_; ++j) { ND[j] = N(j, order_ - n_); }
            // compute the nth derivative using the table of width n_
            for (int jj = 1; jj <= n_; jj++) {
                double saved = 0.0;
                if (ND[0] != 0.0) { saved = ND[0] / (knots_[i_ + order_ - n_ + jj] - knots_[i_]); }
                for (int j = 0; j < n_ - jj + 1; j++) {
                    double Uleft  = knots_[i_ + j + 1];
                    double Uright = knots_[i_ + j + 1 + order_ - n_ + jj];
                    if (ND[j + 1] == 0.0) {
                        ND[j] = (order_ - n_ + jj) * saved;
                        saved = 0.0;
                    } else {
                        double temp = ND[j + 1] / (Uright - Uleft);
                        ND[j] = (order_ - n_ + jj) * (saved - temp);
                        saved = temp;
                    }
                }
            }
            return ND[0];
        }
    };
   public:
    // constructor
    constexpr Spline() = default;
    template <typename KnotsVectorType>
        requires(requires(KnotsVectorType knots, int i) {
                    { knots[i] } -> std::convertible_to<double>;
                    { knots.size() } -> std::convertible_to<std::size_t>;
                })
    Spline(KnotsVectorType&& knots, int i, int order) : i_(i), order_(order) {
        fdapde_assert(
          i >= 0 && order >= 0 && std::cmp_greater_equal(knots.size(), order + 1) && std::cmp_less(i_, knots.size()));
        knots_.reserve(knots.size());
        for (std::size_t i = 0; i < knots.size(); ++i) { knots_.push_back(knots[i]); }
    };

    // non-recursive implementation of spline evaluation, as detailed in "Piegl, L., & Tiller, W. (2012). The NURBS
    // book. Springer Science & Business Media. Algorithm A2.4 pag 74"
    template <typename InputType_>
        requires(internals::is_subscriptable<InputType_, int> || std::is_floating_point_v<InputType_>)
    constexpr Scalar operator()(const InputType_& p_) const {
        double p;
        if constexpr (internals::is_subscriptable<InputType_, int>) {
            p = p_[0];
        } else {
            p = p_;
        }
        int m = knots_.size() - 1;
        std::vector<double> N(order_ + 1, 0.0);
        // special cases
        if ((i_ == 0 && p == knots_[0]) || (i_ == m - order_ - 1 && p == knots_[m])) return 1.0;
        // local property: return 0 if p is outside the range of this basis function
	if (p < knots_[i_] || p >= knots_[i_ + order_ + 1]) return 0.0;
        // initialize 0th degree basis functions
        for (int j = 0; j <= order_; ++j) {
            if (p >= knots_[i_ + j] && p < knots_[i_ + j + 1]) { N[j] = 1.0; }
        }
        // compute triangular table
        for (int k = 1; k <= order_; ++k) {
            double saved = (N[0] == 0.0) ? 0.0 : ((p - knots_[i_]) * N[0]) / (knots_[i_ + k] - knots_[i_]);
            for (int j = 0; j < order_ - k + 1; ++j) {
                double Uleft  = knots_[i_ + j + 1];
                double Uright = knots_[i_ + j + k + 1];
                if (N[j + 1] == 0.0) {
                    N[j] = saved;
                    saved = 0.0;
                } else {
                    double temp = N[j + 1] / (Uright - Uleft);
                    N[j] = saved + (Uright - p) * temp;
                    saved = (p - Uleft) * temp;
                }
            }
        }
        return N[0];
    }
    constexpr Derivative gradient(int n = 1) const { return Derivative(knots_, i_, order_, n); }
    const std::vector<double>& knot_vector() const { return knots_; }
    int order() const { return order_; }
    int knot_id() const { return i_ + order_ - 1; }
    double knot() const { return knots_[i_ + order_ - 1]; }
    constexpr int input_size() const { return StaticInputSize; }
   private:
    std::vector<double> knots_ {};
    int i_ = 0;       // knot index where this basis element is centered
    int order_ = 0;   // B-spline order
};

constexpr auto dx (const Spline& spline) { return spline.gradient(1); }
constexpr auto ddx(const Spline& spline) { return spline.gradient(2); }

}   // namespace fdapde

#endif // __FDAPDE_SPLINE_H__
