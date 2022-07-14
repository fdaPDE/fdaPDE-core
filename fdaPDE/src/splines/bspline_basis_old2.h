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
#include "sp_periodic_bc.h"

namespace fdapde {

int findInterval(const std::vector<double>& u, double x) {
    if(x == u.back()) {
        return static_cast<int>(u.size() - 2);
    }
    for (size_t i = 0; i < u.size() - 1; ++i) {
        if (x >= u[i] && x < u[i + 1]) {
            return static_cast<int>(i);
        }
    }
    return -1; // Return -1 if x is not in any interval
}

template <typename KnotsVectorType>
requires(requires(KnotsVectorType knots, int i) {
            { knots[i] } -> std::convertible_to<double>;
            { knots.size() } -> std::convertible_to<std::size_t>;
        })
std::vector<double> pad_knots(const KnotsVectorType& kn, int degree) {
    int n = kn.size();
    std::vector<double> knots(kn.begin(), kn.end());

    // Check if knots are already padded correctly
    bool is_padded = (n >= 2 * degree + 1);
    for (int i = 0; i < degree + 1 && is_padded; ++i) {
        if (knots[i] != knots[0] || knots[n - 1 - i] != knots[n - 1]) {
            is_padded = false;
        }
    }
    if (is_padded) return knots;

    // If not padded, construct the padded knot vector
    std::vector<double> padded_knots;
    padded_knots.reserve(n + 2 * degree);

    padded_knots.insert(padded_knots.end(), degree, knots[0]);
    padded_knots.insert(padded_knots.end(), knots.begin(), knots.end());
    padded_knots.insert(padded_knots.end(), degree, knots[n - 1]);

    return padded_knots;
}

// given vector of knots u_1, u_2, ..., u_N, this class represents the set of N + degree - 1 spline basis functions
// {l_1(x), l_2(x), ..., l_{N + degree - 1}(x)} centered at knots u_1, u_2, ..., u_N
class BSplineBasis {
   private:
    int degree_;
    std::vector<Spline> basis_ {};
    std::vector<double> knots_ {};
    std::vector<double> unique_knots_ {};
    bool periodicity_ = false;
    int n_basis_ = 0;

   public:
    static constexpr int StaticInputSize = 1;
    static constexpr int Order = Dynamic;
    // constructors
    constexpr BSplineBasis() : degree_(0) { }
    // constructor from user defined knot vector
    
    template <typename KnotsVectorType>
    requires(requires(KnotsVectorType knots, int i) {
                { knots[i] } -> std::convertible_to<double>;
                { knots.size() } -> std::convertible_to<std::size_t>;
            })
    BSplineBasis(KnotsVectorType&& knots, int degree, bool periodicity = false)
        : degree_(degree), knots_(pad_knots(knots, degree)), periodicity_(periodicity) {
        int n = knots.size();
        basis_.reserve(n - degree_ - 1);

        for (int i = 0; i < n - degree_ - 1; ++i) {
            basis_.emplace_back(knots_, i, degree_);
        }

        // compute unique knots
        std::set<double> unique_knots_set(knots_.begin(), knots_.end());
        unique_knots_.assign(unique_knots_set.begin(), unique_knots_set.end());
        // check if the knots are open and uniform
        if(periodicity_) {
            n_basis_ = knots_.size() - 2* degree_ - 1;
        }
        else {
            n_basis_ = knots_.size() - degree_ - 1;
        }

    }
    // Constructor from geometric interval (no repeated knots)
    BSplineBasis(const Triangulation<1, 1>& interval, int degree, bool periodicity = false)
        : degree_(degree), periodicity_(periodicity) {
        Eigen::VectorXd knots = interval.nodes();
        fdapde_assert(std::is_sorted(knots.begin(), knots.end(), std::less_equal<double>()));
        knots_ = pad_knots(std::vector<double>(knots.data(), knots.data() + knots.size()), degree);
        int n = knots_.size();
        basis_.reserve(n - degree_ - 1);
        for (int i = 0; i < n - degree_ - 1; ++i) {
            basis_.emplace_back(knots_, i, degree_);
        }
    }

    // Algorithm A2.1 from NURBS book
    int find_span(double x, int n = - 1) const {
        if (n == -1) n = knots_.size() - degree_ - 1;
        if (x == knots_.back()) return n - 1;
        int low = degree_, high = n, mid;
        while (low < high - 1) {
            mid = (low + high) / 2;
            (x < knots_[mid]) ? high = mid : low = mid;
        }
        return low;
    }

    std::vector<double> evaluate_basis(double x, bool pad = true) const {
        if (!periodicity_) {
            auto N = evaluate_basis_(x);
            int i = find_span(x);
            std::vector<double> padded_N(knots_.size() - degree_ - 1, 0.0);
            int start_index = i - degree_;
            for (int j = 0; j <= degree_; j++) {
                padded_N[start_index + j] = N[j];
            }
            return pad ? padded_N : N;
        }

        int idx = findInterval(unique_knots_, x);
        std::vector<double> local = evaluate_basis_(x);
        Eigen::Map<const Eigen::VectorXd> N_vec(local.data(), local.size());
        auto T_per = internals::bs_periodic_transformation(degree_, idx, knots_.size() - 2 * degree_ - 1);
        Eigen::VectorXd N_per = T_per * N_vec;

        std::vector<double> transformed(N_per.data(), N_per.data() + N_per.size());
        return pad ? pad_periodic_values(transformed, x) : transformed;
    }

    std::vector<double> evaluate_der_basis(double x, int n = 1, bool pad = true) const {
        if (!periodicity_) {
            auto N = evaluate_der_basis_(x,n);
            int i = find_span(x);
            std::vector<double> padded_N(knots_.size() - degree_ - 1, 0.0);
            int start_index = i - degree_;
            for (int j = 0; j <= degree_; j++) {
                padded_N[start_index + j] = N[j];
            }
            return pad ? padded_N : N;
        }

        int idx = findInterval(unique_knots_, x);
        std::vector<double> local = evaluate_der_basis_(x, n);
        Eigen::Map<const Eigen::VectorXd> N_vec(local.data(), local.size());
        auto T_per = internals::bs_periodic_transformation(degree_, idx, knots_.size() - 2 * degree_ - 1);
        Eigen::VectorXd N_per = T_per * N_vec;

        std::vector<double> transformed(N_per.data(), N_per.data() + N_per.size());
        return pad ? pad_periodic_values(transformed, x) : transformed;
    }
        

    private:
    //Algorithm A2. 2 from NURBS book pag. 70 computes all the nonvanishing
    //basis functions and stores them in the array N [0] , ... ,N [p]
    // padded with zeros
    // Evaluate basis functions at x
    std::vector<double> evaluate_basis_(double x) const {
        std::vector<double> N(degree_ + 1, 0.0);
        std::vector<double> left(degree_ + 1), right(degree_ + 1);
        N[0] = 1.0;

        int i = find_span(x);

        for (int j = 1; j <= degree_; ++j) {
            left[j] = x - knots_[i + 1 - j];
            right[j] = knots_[i + j] - x;
            double saved = 0.0;

            for (int r = 0; r < j; ++r) {
                double denom = right[r + 1] + left[j - r];
                double temp = (denom == 0) ? 0.0 : N[r] / denom;
                N[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            N[j] = saved;
        }

        return N;
    }

    // A2.3 Evaluate the nth derivative of B-spline basis functions at x, padded with zeros
    std::vector<double> evaluate_der_basis_(double x, int n=1) const {
        // Degree (p) and knot vector (U) from the class

        int i = find_span(x);

        // Output for derivatives
        std::vector<std::vector<double>> ders(n + 1, std::vector<double>(degree_ + 1, 0.0));

        // Temporary arrays
        std::vector<std::vector<double>> ndu(degree_ + 1, std::vector<double>(degree_ + 1, 0.0));
        std::vector<double> left(degree_ + 1, 0.0);
        std::vector<double> right(degree_ + 1, 0.0);

        // Compute basis functions and differences
        ndu[0][0] = 1.0;
        for (int j = 1; j <= degree_; ++j) {
            left[j] = x - knots_[i + 1 - j];
            right[j] = knots_[i + j] - x;

            double saved = 0.0;
            for (int r = 0; r < j; ++r) {
                // Lower triangle
                ndu[j][r] = right[r + 1] + left[j - r];
                double temp = ndu[r][j - 1] / ndu[j][r];

                // Upper triangle
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        // Load the basis functions
        for (int j = 0; j <= degree_; ++j) {
            ders[0][j] = ndu[j][degree_];
        }

        // Compute derivatives
        std::vector<std::vector<double>> a(2, std::vector<double>(degree_ + 1, 0.0));
        for (int r = 0; r <= degree_; ++r) {
            int s1 = 0, s2 = 1; // Alternate rows in array a
            a[0][0] = 1.0;

            for (int k = 1; k <= n; ++k) {
                double d = 0.0;
                int rk = r - k, pk = degree_ - k;

                if (r >= k) {
                    a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                    d = a[s2][0] * ndu[rk][pk];
                }

                int j1 = (rk >= -1) ? 1 : -rk;
                int j2 = (r - 1 <= pk) ? k - 1 : degree_ - r;

                for (int j = j1; j <= j2; ++j) {
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                    d += a[s2][j] * ndu[rk + j][pk];
                }

                if (r <= pk) {
                    a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                    d += a[s2][k] * ndu[r][pk];
                }

                ders[k][r] = d;

                // Swap rows
                std::swap(s1, s2);
            }
        }
        
        // Multiply by the correct factors
        auto r = degree_;
        for (int k = 1; k <= n; ++k) {
            for (int j = 0; j <= degree_; ++j) {
                ders[k][j] *= r;
            }
            r *= (degree_ - k);
        }

        return ders[n];
    }

    std::vector<double> pad_periodic_values(const std::vector<double>& local, double x) const {
        int span = find_span(x);
        int n_total = knots_.size() - degree_ - 1;
        int n_unique = n_basis();
        std::vector<double> padded(n_total, 0.0);
        for (int j = 0; j <= degree_; ++j) {
            int original_idx = span - degree_ + j;
            int wrapped_idx = (original_idx % n_total + n_total) % n_total;
            padded[wrapped_idx] += local[j];
    
            int mirror_idx = (wrapped_idx < n_unique) ? wrapped_idx + n_unique : wrapped_idx - n_unique;
            if (mirror_idx >= 0 && mirror_idx < n_total) {
                padded[mirror_idx] += local[j];
            }
        }
        return padded;
    }
    
    public:
    // getters
    constexpr const Spline& operator[](int i) const { return basis_[i]; }
    constexpr int size() const { return basis_.size(); }
    constexpr const std::vector<double>& knots_vector() const { return knots_; }
    int n_knots() const { return knots_.size(); }
    int degree() const { return degree_; }
    bool periodicity() const { return periodicity_;}
    int n_basis() const { return n_basis_; }
};



} // namespace fdapde

#endif // __FDAPDE_BSPLINE_BASIS_H__
