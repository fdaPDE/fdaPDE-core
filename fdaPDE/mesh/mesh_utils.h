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

    // affine transformations from cartesian to barycentric coordinates and viceversa
    SMatrix<N, M-1> J_;       // [J_]_ij = (coords_[j][i] - coords_[0][i])
    SMatrix<M-1, N> inv_J_;   // J^{-1} (Penrose pseudo-inverse for manifold elements)
    double measure_ = 0;      // measure of the facet
   public:
    Facet(std::size_t ID, const std::array<int, M>& node_ids, const std::array<SVector<N>, M>& coords,
	  const std::vector<int>& elements, bool boundary) :
      ID_(ID), node_ids_(node_ids), coords_(coords), elements_(elements), boundary_(boundary) {
        // precompute barycentric coordinate matrix for fast access
        // use first point as reference
        SVector<N> ref = coords_[0];
        for (std::size_t j = 0; j < (M-1); ++j) { J_.col(j) = coords_[j + 1] - ref; }

        // precompute and cache inverse of barycentric matrix
        if constexpr (N == M-1) {
            inv_J_ = J_.inverse();
        } else {
            // for manifold domains compute the generalized Penrose inverse
            inv_J_ = (J_.transpose() * J_).inverse() * J_.transpose();
        }
        // precompute element measure
        if constexpr ((M-1) == N) {   // non-manifold case
            measure_ = std::abs(J_.determinant()) / (ct_factorial(M-1));
        } else {
            if constexpr (M == 3)   // surface element, compute area of 3D triangle
                measure_ = 0.5 * J_.col(0).cross(J_.col(1)).norm();
            if constexpr (M == 2)   // network element, compute length of 2D segment
                measure_ = J_.col(0).norm();
        }
    }

    // getters
    std::size_t ID() const { return ID_; }
    const std::array<SVector<N>, M>& coords() const { return coords_; }
    const std::vector<int>& adjacent_elements() const { return elements_; }
    const std::array<int, M>& node_ids() const { return node_ids_; }
    bool on_boundary() const { return boundary_; }
    Eigen::Matrix<double, N, M-1> barycentric_matrix() const { return J_; }
    Eigen::Matrix<double, M-1, N> inv_barycentric_matrix() const { return inv_J_; }
    double measure() const { return measure_; }   // measure of the facet

    // computes midpoint of the element
    SVector<N> mid_point() const {
        // The center of gravity of a facet has all its barycentric coordinates equal to 1/(M).
        // The midpoint of an element can be computed by mapping it to a cartesian reference system
        SVector<M-1> barycentric_mid_point;
        barycentric_mid_point.fill(1.0 / (M));   // avoid implicit conversion to int
        return J_ * barycentric_mid_point + coords_[0];
    }
};


template <int N> class Facet<1,N> {
   private:
    std::size_t ID_;
    std::vector<int> elements_;          // elements insisting on the facet
    std::array<int, 1> node_ids_;        // ID of nodes composing the facet
    std::array<SVector<N>, 1> coords_;   // pyhisical coordinates of the nodes composing the facet
    bool boundary_; // asserted true if the edge is on boundary, i.e. all its nodes are boundary nodes

   public:
    Facet(std::size_t ID, const std::array<int, 1>& node_ids, const std::array<SVector<N>, 1>& coords,
	  const std::vector<int>& elements, bool boundary) :
      ID_(ID), node_ids_(node_ids), coords_(coords), elements_(elements), boundary_(boundary) { };

    // getters
    std::size_t ID() const { return ID_; }
    const std::array<SVector<N>, 1>& coords() const { return coords_; }
    const std::vector<int>& adjacent_elements() const { return elements_; }
    const std::array<int, 1>& node_ids() const { return node_ids_; }
    bool on_boundary() const { return boundary_; }

    // computes midpoint of the element
    SVector<N> mid_point() const {
        return coords_[0];
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __MESH_UTILS_H__
