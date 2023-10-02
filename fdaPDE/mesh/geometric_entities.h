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

#ifndef __GEOMETRIC_ENTITIES_H__
#define __GEOMETRIC_ENTITIES_H__

#include "../utils/symbols.h"
#include "../utils/combinatorics.h"

namespace fdapde {
namespace core {

// number of degrees of freedom associated to an M-dimensional element of degree R
constexpr int ct_nnodes(const int M, const int R) {
    return ct_factorial(M + R) / (ct_factorial(M) * ct_factorial(R));
}
// number of vertices of an M-dimensional element
constexpr int ct_nvertices(const int M) { return M + 1; }
// number of edges of an M-dimensional element
constexpr int ct_nfacets(const int M) { return (M * (M + 1)) / 2; }
// number of faces of an M-dimensional element
constexpr int ct_nneighbors(const int M) { return (M == 1) ? fdapde::Dynamic : (M + 1); }
  
template <int M, int N> class Facet {
   private:
    std::size_t ID_;
    std::array<int, 2> elements_;        // elements insisting on the edge
    std::array<int, M> node_ids_;        // ID of nodes composing the edge
    std::array<SVector<N>, M> coords_;   // pyhisical coordinates of the nodes composing the edge
    bool boundary_; // asserted true if the edge is on boundary, i.e. all its nodes are boundary nodes
   public:
    Facet(std::size_t ID, const std::array<int, M>& node_ids, const std::array<SVector<N>, M>& coords,
	  const std::array<int, 2>& elements, bool boundary) :
      ID_(ID), node_ids_(node_ids), coords_(coords), elements_(elements), boundary_(boundary) {};

    // getters
    std::size_t ID() const { return ID_; }
    const std::array<SVector<N>, M>& coords() const { return coords_; }
    const std::array<int, 2>& adjacent_elements() const { return elements_; }
    const std::array<int, M>& node_ids() const { return node_ids_; }
    bool on_boundary() const { return boundary_; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __GEOMETRIC_ENTITIES_H__
