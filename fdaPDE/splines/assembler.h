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

#ifndef __TIME_ASSEMBLER_H__
#define __TIME_ASSEMBLER_H__

#include "../utils/symbols.h"
#include "../finite_elements/integration/integrator.h"
#include "../finite_elements/integration/integrator_tables.h"
#include "spline_basis.h"

namespace fdapde {
namespace core {

  // base class providing time discretization matrices. B is the type of basis used for discretizing the time dimension
  // this is a specialized assembly loop for 1D problems which do not require all the machinery put in place in the FEM assembler
  template <typename B> class TimeAssembler;
  
  // specialization of TimeAssembler for a spline basis of order R
  template <unsigned int R>
  class TimeAssembler<SplineBasis<R>> {
  private:
    const DVector<double>& time_; // time mesh
    SplineBasis<R> basis_{};    
    IntegratorTable<1, 3, GaussLegendre> quad_rule_;
    
    // if the operator is symmetric returns i, otherwise the inner assembly loop must run over the whole set of basis funcitons
    template <typename E>
    constexpr std::size_t compute_inner_loop_limit(std::size_t i, std::size_t M) const {
      if constexpr(E::is_symmetric) return i;
      else return M;
    }
  public:
    // constructor
    TimeAssembler(const DVector<double>& time) : time_(time), basis_(time_) {};

    // computes the discretization of E
    template <typename E>
    SpMatrix<double> assemble(const E& f) const;
  };

  template <unsigned int R>
  template <typename E>
  SpMatrix<double> TimeAssembler<SplineBasis<R>>::assemble(const E& op) const {
    // compute result dimensions
    std::size_t M = basis_.size();
    // resize result matrix
    SpMatrix<double> discretizationMatrix;
    discretizationMatrix.resize(M,M);
    // prepare triplet list for sparse matrix construction
    std::vector<fdaPDE::Triplet<double>> tripletList;
    tripletList.reserve(M*M);
    
    // start assembly loop (exploit local support of spline basis)
    for(std::size_t i = 0; i < M; ++i){
      for(std::size_t j = 0; j <= compute_inner_loop_limit<E>(i,M); ++j){
	// develop integrand field
	auto f = op.integrate(basis_[i], basis_[j]);
	// perform integration of f over [knots[j], knots[i+R+1]]
        double value = 0;
	for(std::size_t k = j; k <= i+R; ++k){
	  value += integrate(basis_.knots()[k], basis_.knots()[k+1], f);
	}
	tripletList.emplace_back(i,j, value); // store computed integral
      }
    }
    // finalize construction
    discretizationMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
    discretizationMatrix.makeCompressed();
    if constexpr(E::is_symmetric)
      return discretizationMatrix.selfadjointView<Eigen::Lower>();
    else
      return discretizationMatrix;
  };
  
  // functor for the computation of the (i,j)-th element of the time mass discretization matrix \phi_i*\phi_j
  template <typename B> class TimeMass;  
  // partial specialization for SplineBasis case.
  template <unsigned int R>
  struct TimeMass<SplineBasis<R>> {
    static constexpr bool is_symmetric = true;
    // provide \phi_i * \phi_j
    auto integrate(const Spline<R>& phi_i, const Spline<R>& phi_j) const {
      return phi_i * phi_j;
    };
  };

  // functor for the computation of the (i,j)-th element of the time penalty matrix (\phi_i)_tt * (\phi_j)_tt
  template <typename B> class TimePenalty;
  // partial specialization for SplineBasis case.
  template <unsigned int R>
  struct TimePenalty<SplineBasis<R>> {
    static constexpr bool is_symmetric = true;
    // provide (\phi_i)_tt * (\phi_j)_tt
    auto integrate(const Spline<R>& phi_i, const Spline<R>& phi_j) const {
      return phi_i.template derive<2>() * phi_j.template derive<2>();
    };
  };
  
}}

#endif // __TIME_ASSEMBLER_H__
