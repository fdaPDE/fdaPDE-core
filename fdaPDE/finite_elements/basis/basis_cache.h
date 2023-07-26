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

#ifndef __BASIS_CACHE_H__
#define __BASIS_CACHE_H__

namespace fdapde {
namespace core {
  
// a functor representing a finite element over a pyhsical element e written as function of a reference basis B
template <unsigned int M, unsigned int N, unsigned int R, typename B> class FiniteElement {
   private:
    typedef typename B::ElementType ElementType;

    std::size_t node_;                     // if the element is written as \phi_i, i is the value of node_
    const Element<M, N, R>& e_;            // physical element over which the basis is defined
    const ElementType& reference_basis_;   // basis wrt which this finite element is written
   public:
    FiniteElement() = default;
    FiniteElement(std::size_t node, const Element<M, N, R>& e, const ElementType& reference_basis) :
        node_(node), e_(e), reference_basis_(reference_basis) {};

    // call operator
    inline double operator()(const SVector<N>& x) const {
        // map x into reference element
        SVector<N> p = e_.inv_barycentric_matrix() * (x - e_.coords()[0]);
        return reference_basis_(p);   // evaluate reference basis at p
    }
    std::size_t node() const { return node_; }
};

// the i-th element of this cache returns the set of basis functions built over the i-th element of the mesh
template <unsigned int M, unsigned int N, unsigned int R, typename B>
using BasisCache = std::vector<std::vector<FiniteElement<M, N, R, B>>>;

}   // namespace core
}   // namespace fdapde

#endif   // __BASIS_CACHE_H__
