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

#include <optional>

#include "../utils/symbols.h"
#include "hyperplane.h"
#include "utils.h"

namespace fdapde {
namespace core {

template <typename MeshType> class Element {
   public:
    static constexpr int local_dim = MeshType::local_dim;
    static constexpr int embed_dim = MeshType::embed_dim;
    static constexpr int n_vertices = ct_nvertices(local_dim);
    using bbox_type = std::pair<SVector<embed_dim>, SVector<embed_dim>>;
    // constructor
    Element() = default;
    Element(int id, const MeshType* mesh) : id_(id), mesh_(mesh), boundary_(false) {
        for (int j = 0; j < n_vertices; ++j) { boundary_ |= mesh_->is_on_boundary(mesh_->elements()(id_, j)); }
    }
    // getters
    SMatrix<embed_dim, n_vertices> coords() const {
        SMatrix<embed_dim, n_vertices> coords {};
        for (int j = 0; j < n_vertices; ++j) {
            coords.col(j) = mesh_->node(mesh_->elements()(id_, j));   // physical coordinate of the node
        }
        return coords;
    }
    SVector<embed_dim> coord(int vertex) const {   // coordinate of the i-th vertex of this element
        return mesh_->node(mesh_->elements()(id_, vertex));
    }
    auto neighbors() const { return mesh_->neighbors().row(id_); }
    int ID() const { return id_; }
    const SMatrix<embed_dim, local_dim>& J() const {
        if (!J_.has_value()) {
            J_ = SMatrix<embed_dim, local_dim>::Zero();
            for (int j = 0; j < local_dim; ++j) { J_->col(j) = coord(j + 1) - coord(0); }
        }
        return J_.value();
    }
    const SMatrix<local_dim, embed_dim>& invJ() const {
        if (!invJ_.has_value()) {
            if constexpr (embed_dim == local_dim) {
                invJ_ = J().inverse();
            } else {   // generalized Penrose inverse for manifold eleents
                invJ_ = (J().transpose() * J()).inverse() * J().transpose();
            }
        }
        return invJ_.value();
    }
    auto node_ids() const { return mesh_->elements().row(id_); }
    double measure() const {
        if (measure_ < 0) {
            if constexpr (embed_dim == local_dim) {
                measure_ = std::abs(J().determinant()) / (ct_factorial(local_dim));
            } else {
                if constexpr (local_dim == 2) measure_ = 0.5 * J().col(0).cross(J().col(1)).norm();
                if constexpr (local_dim == 1) measure_ = J().col(0).norm();
            }
        }
        return measure_;
    }
    // check if x is contained in the element (checks for positiveness of barycentric coordinates of x)
    bool contains(const SVector<embed_dim>& x) const {
        if constexpr (local_dim != embed_dim) {   // check if the point is contained in the hyperplane through this
            if (hyperplane().distance(x) > tolerance_) return false;
        }
        return (to_barycentric_coords(x).array() >= -tolerance_).all();
    }
    // move x into the barycentric reference system of this element
    SVector<local_dim + 1> to_barycentric_coords(const SVector<embed_dim>& x) const {
        SVector<local_dim> z = invJ() * (x - coord(0));
        SVector<local_dim + 1> result;
        result << SVector<1>(1 - z.sum()), z;
        return result;
    }
    // computes midpoint of the element
    SVector<embed_dim> mid_point() const {
        SVector<local_dim> barycenter;
        barycenter.fill(1.0 / (local_dim + 1));   // the barycenter has all its barycentric coordinates equal to 1/(M+1)
        return J() * barycenter + coord(0);
    }
    // compute the smallest rectangle containing this element
    bbox_type bounding_box() const {
        return std::make_pair(coords().rowwise().minCoeff(), coords().rowwise().maxCoeff());
    }
    bool is_on_boundary() const { return boundary_; }
    HyperPlane<local_dim, embed_dim> hyperplane() const { return HyperPlane<local_dim, embed_dim>(coords()); }
    operator bool() const { return mesh_ != nullptr; }
   private:
    static constexpr double tolerance_ = 10 * std::numeric_limits<double>::epsilon();
    int id_ = 0;   // ID of this element in the physical mesh
    const MeshType* mesh_ = nullptr;
    bool boundary_ = false;   // true if the element has at least one vertex on the boundary
    mutable double measure_ = -1;                                 // element's measure
    mutable std::optional<SMatrix<embed_dim, local_dim>> J_;      // [J_]_ij = (coords_(j,i) - coords_(0,i))
    mutable std::optional<SMatrix<local_dim, embed_dim>> invJ_;   // J^{-1} (Penrose pseudo-inverse for manifold)
};

}   // namespace core
}   // namespace fdapde

#endif   // __ELEMENT_H__
