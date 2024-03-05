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

#ifndef __ELEMENT_H__
#define __ELEMENT_H__

#include <array>

#include "../utils/symbols.h"
#include "hyperplane.h"
#include "mesh_utils.h"

namespace fdapde {
namespace core {
  
// A mesh element. M: tangent space dimension, N: embedding dimension
template <int M, int N> class Element {
   public:
    using neigh_container = std::conditional_t<is_network<M, N>::value, std::vector<int>, std::array<int, M + 1>>;
    using bbox_type       = std::pair<SVector<N>, SVector<N>>;
    static constexpr double tolerance_ = 10 * std::numeric_limits<double>::epsilon();
    // compile time constants
    enum {
        n_vertices = ct_nvertices(M),
        local_dimension = M,
        embedding_dimension = N
    };
    // constructor
    Element() = default;
    Element(int id, const std::array<int, n_vertices>& node_ids, const SMatrix<N, n_vertices>& coords,
        const neigh_container& neighbors, bool boundary) :
        id_(id), node_ids_(node_ids), coords_(coords), neighbors_(neighbors), boundary_(boundary) {
        // precompute barycentric coordinate matrix for fast access, use first point as reference
        for (std::size_t j = 0; j < M; ++j) { J_.col(j) = coords_.col(j + 1) - coords_.col(0); }
        // precompute and cache inverse of barycentric matrix and element's measure
        if constexpr (N == M) {
            inv_J_ = J_.inverse();
            measure_ = std::abs(J_.determinant()) / (ct_factorial(M));
        } else {
            inv_J_ = (J_.transpose() * J_).inverse() * J_.transpose();   // generalized Penrose inverse
            if constexpr (M == 2)
                measure_ = 0.5 * J_.col(0).cross(J_.col(1)).norm();
            if constexpr (M == 1)
                measure_ = J_.col(0).norm();
        }
    }
    // getters
    const SMatrix<N, n_vertices>& coords() const { return coords_; }
    auto coord(int vertex) const { return coords_.col(vertex); }   // coordinate of i-th vertex
    const neigh_container& neighbors() const { return neighbors_; }
    int ID() const { return id_; }
    SMatrix<N, M> barycentric_matrix() const { return J_; }
    SMatrix<M, N> inv_barycentric_matrix() const { return inv_J_; }
    std::array<int, n_vertices> node_ids() const { return node_ids_; }
    double measure() const { return measure_; }   // measure of the element
  
    // check if x is contained in the element (checks for positiveness of barycentric coordinates of x)
    bool contains(const SVector<N>& x) const {
        if constexpr (N != M) {
            // check if the point is contained in the affine space spanned by the element
            if (hyperplane().distance(x) > tolerance_) return false;
        }
        return (to_barycentric_coords(x).array() >= -tolerance_).all();
    }
    // move x into the barycentric reference system of this element
    SVector<M + 1> to_barycentric_coords(const SVector<N>& x) const {
        SVector<M> z = inv_J_ * (x - coords_.col(0));
        SVector<M + 1> result;
        result << SVector<1>(1 - z.sum()), z;
        return result;
    }
    // computes midpoint of the element
    SVector<N> mid_point() const {
        SVector<M> barycenter;
        barycenter.fill(1.0 / (M + 1));   // the barycenter has all its barycentric coordinates equal to 1/(M+1).
        return J_ * barycenter + coords_.col(0);
    }
    // compute the smallest rectangle containing this element
    bbox_type bounding_box() const {
        return std::make_pair(coords_.rowwise().minCoeff(), coords_.rowwise().maxCoeff());
    }
    bool is_on_boundary() const { return boundary_; }   // true if the element has at least one node on the boundary
    // hyperplane passing throught this element
    HyperPlane<M, N> hyperplane() const {
        return HyperPlane<M, N>(coords_);
    }
   private:
    int id_ = 0;
    std::array<int, n_vertices> node_ids_ {};   // vertices ids
    SMatrix<N, n_vertices> coords_ {};          // vertices coordinates (1-1 mapped with node_ids_)
    neigh_container neighbors_ {};
    bool boundary_;         // true if the element has at least one vertex on the boundary
    double measure_ = 0;    // element's measure
    SMatrix<N, M> J_;       // [J_]_ij = (coords_(j,i) - coords_(0,i))
    SMatrix<M, N> inv_J_;   // J^{-1} (Penrose pseudo-inverse for manifold elements)
};

}   // namespace core
}   // namespace fdapde

#endif   // __ELEMENT_H__
