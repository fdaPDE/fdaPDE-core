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

#ifndef __MESH_UTILS_H__
#define __MESH_UTILS_H__

#include "../utils/symbols.h"
#include "../utils/combinatorics.h"

namespace fdapde {
namespace core {

// trait to detect if a mesh is a manifold
template <int M, int N> struct is_manifold {
    static constexpr bool value = (M != N);
};

// macro for the definition of mesh type detection
#define DEFINE_MESH_TYPE_DETECTION_TRAIT(M_, N_, name)                                                                 \
    template <int M, int N> struct is_##name {                                                                         \
        static constexpr bool value =                                                                                  \
          std::conditional<(M == M_ && N == N_), std::true_type, std::false_type>::type::value;                        \
    };

DEFINE_MESH_TYPE_DETECTION_TRAIT(1, 2, network);   // is_network<M, N>
DEFINE_MESH_TYPE_DETECTION_TRAIT(2, 2, 2d);        // is_2d<M, N>
DEFINE_MESH_TYPE_DETECTION_TRAIT(2, 3, surface);   // is_surface<M, N>
DEFINE_MESH_TYPE_DETECTION_TRAIT(3, 3, 3d);        // is_3d<M, N>
  
// number of degrees of freedom associated to an M-dimensional element of degree R
constexpr int ct_nnodes(const int M, const int R) {
    return ct_factorial(M + R) / (ct_factorial(M) * ct_factorial(R));
}
constexpr int ct_nvertices(const int M) { return M + 1; }
constexpr int ct_nfacets(const int M) { return (M * (M + 1)) / 2; }
constexpr int ct_nneighbors(const int M) { return (M == 1) ? fdapde::Dynamic : (M + 1); }

template <int M, int N> class Facet {
   private:
    std::size_t ID_;
    std::vector<int> elements_;          // elements insisting on the facet
    std::array<int, M> node_ids_;        // ID of nodes composing the facet
    std::array<SVector<N>, M> coords_;   // pyhisical coordinates of the nodes composing the facet
    bool boundary_; // asserted true if the edge is on boundary, i.e. all its nodes are boundary nodes
   public:
    Facet(std::size_t ID, const std::array<int, M>& node_ids, const std::array<SVector<N>, M>& coords,
	  const std::vector<int>& elements, bool boundary) :
      ID_(ID), node_ids_(node_ids), coords_(coords), elements_(elements), boundary_(boundary) {};

    // getters
    std::size_t ID() const { return ID_; }
    const std::array<SVector<N>, M>& coords() const { return coords_; }
    const std::vector<int>& adjacent_elements() const { return elements_; }
    const std::array<int, M>& node_ids() const { return node_ids_; }
    bool on_boundary() const { return boundary_; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __MESH_UTILS_H__
