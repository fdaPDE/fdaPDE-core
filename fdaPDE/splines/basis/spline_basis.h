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
#include "../../utils/integration/integrator_tables.h"
#include "spline.h"

namespace fdapde {
namespace core {

// a spline basis of order R built over a given set of knots
template <int R> class SplineBasis {
   private:
    DVector<double> knots_ {};   // vector of knots
    std::vector<Spline<R>> basis_ {};
   public:
    static constexpr std::size_t order = R;
    typedef Spline<R> ElementType;
    typedef Integrator<SPLINE, 1, order> Quadrature;
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

    // returns the matrix \Phi of basis functions evaluations at the given locations
    template <template <typename> typename EvaluationPolicy>
    std::pair<SpMatrix<double>, DVector<double>> eval(const DVector<double>& locs) const {
        return EvaluationPolicy<SplineBasis<R>>::eval(*this, locs, basis_.size());
    }
    const Spline<R>& operator[](std::size_t i) const { return basis_[i]; }
    int size() const { return basis_.size(); }
    const DVector<double>& knots() const { return knots_; }
    // given a coefficient vector c \in \mathbb{R}^size_, evaluates the corresponding basis expansion at locs
    DVector<double> operator()(const DVector<double>& c, const DVector<double>& locs) const {
        fdapde_assert(c.rows() == size() && locs.cols() != 0);
        DVector<double> result = DVector<double>::Zero(locs.rows());
        for (std::size_t i = 0; i < locs.rows(); ++i) {
            // evaluate basis expansion \sum_{i=1}^size_ c_i \phi_i(x) at p
            SVector<1> p(locs[i]);
            for (std::size_t h = 0; h < basis_.size(); ++h) { result[i] += c[h] * basis_[h](p); }
        }
        return result;
    }
};

template <int R> struct pointwise_evaluation<SplineBasis<R>> {
    using BasisType = SplineBasis<R>;
    // computes a matrix \Phi such that [\Phi]_{ij} = \phi_j(t_i)
    static std::pair<SpMatrix<double>, DVector<double>>
    eval(const BasisType& basis, const DVector<double>& locs, std::size_t n_basis) {
        fdapde_assert(locs.size() != 0);
        // preallocate space
        SpMatrix<double> Phi(locs.rows(), n_basis);
        std::vector<fdapde::Triplet<double>> triplet_list;
        triplet_list.reserve(locs.rows() * (R + 1));
        // build \Phi matrix
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < locs.rows(); ++j) { triplet_list.emplace_back(j, i, basis[i](SVector<1>(locs[j]))); }
        }
        // finalize construction
        Phi.setFromTriplets(triplet_list.begin(), triplet_list.end());
        Phi.prune(0.0);   // remove zeros
        Phi.makeCompressed();
        return std::pair(std::move(Phi), DVector<double>::Ones(locs.rows()));;
    }
};

  template <int R> struct areal_evaluation<SplineBasis<R>> {
    using BasisType = SplineBasis<R>;
    // computes a matrix \Phi such that [\Phi]_{ij} = \phi_j(t_i)
    static std::pair<SpMatrix<double>, DVector<double>>
    eval(const BasisType& basis, const DVector<double>& locs, std::size_t n_basis) {
      // TODO
      return std::make_pair(SpMatrix<double>{}, DVector<double>::Ones(locs.rows()));
    }
};
  
}   // namespace core
}   // namespace fdapde

#endif   // __SPLINE_BASIS_H__
