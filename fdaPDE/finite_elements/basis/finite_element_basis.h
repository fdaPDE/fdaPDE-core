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

#ifndef __FINITE_ELEMENT_BASIS_H__
#define __FINITE_ELEMENT_BASIS_H__

#include "../../mesh/element.h"
#include "../../utils/symbols.h"

namespace fdapde {
namespace core {

// Given an element e, a point x \in e and a functor F, this functor computes F(P(x)), where P(.) is the bijection
// which maps e into the reference M-dimensional element \hat e. F is intended to be a basis function over \hat e
template <int M, int N, typename F> class FiniteElement {
   private:
    const Element<M, N>& e_;
    const F& f_;
   public:
    FiniteElement(const Element<M, N>& e, const F& f) : e_(e), f_(f) {};

    // computes F(P(x))
    inline double operator()(const SVector<N>& x) const {
        SVector<M> p = e_.inv_barycentric_matrix() * (x - e_.coords()[0]);
        return f_(p);
    }
};

// A finite element basis over a triangulation T made by elements of type B
template <typename B> class FiniteElementBasis {
   public:
    typedef B BasisType;
    FiniteElementBasis() = default;

    // returns the j-th basis function over e. Basis ordering follows the one defined on the reference element
    template <int N, int M>
    FiniteElement<M, N, typename B::ElementType> operator()(const Element<M, N>& e, std::size_t j) const {
        return FiniteElement<M, N, typename B::ElementType>(e, ref_basis_[j]);
    }
   private:
    BasisType ref_basis_ {};   // reference basis over M-dimensional simplex
};

}   // namespace core
}   // namespace fdapde

#endif   // __FINITE_ELEMENT_BASIS_H__
