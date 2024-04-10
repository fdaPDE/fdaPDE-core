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

#ifndef __SIMPLEX_H__
#define __SIMPLEX_H__

#include <optional>

#include "../utils/symbols.h"
#include "hyperplane.h"
#include "utils.h"

namespace fdapde {
namespace core {

// The convex-hull of EmbedDim_ + 1 points in \mathbb{R}^EmbedDim.
// A point (Order 0), line (Order 1), triangle (Order 2), tetrahedron (Order 3) embedded in \mathbb{R}^EmbedDim
template <int Order_, int EmbedDim_> class Simplex {
    static_assert(EmbedDim_ != 0 && Order_ <= 3);
   public:
    static constexpr int local_dim = Order_;
    static constexpr int embed_dim = EmbedDim_;
    static constexpr int n_vertices = ct_nvertices(Order_);
    static constexpr int n_edges = ct_nedges(Order_);
    static constexpr int n_faces = Order_ + 1;
    static constexpr int n_vertices_per_face = Order_;
    using FaceType = Simplex<Order_ - 1, EmbedDim_>;

    Simplex() = default;
    explicit Simplex(const SMatrix<embed_dim, Order_ + 1>& coords) : coords_(coords) { initialize(); }
    // unit simplex constructor
    static Simplex<Order_, EmbedDim_> Unit() {
        SMatrix<embed_dim, Order_ + 1> coords;
        coords.setZero();
        for (int i = 0; i < embed_dim; ++i) coords(i, i + 1) = 1;
        return Simplex(coords);
    }

    // getters
    SVector<embed_dim> vertex(int v) const { return coords_.col(v); }
    SVector<embed_dim> operator[](int v) const { return coords_.col(v); }
    const SMatrix<embed_dim, n_vertices>& vertices() const { return coords_; }
    const SMatrix<embed_dim, local_dim>& J() const { return J_; }
    const SMatrix<local_dim, embed_dim>& invJ() const { return invJ_; }
    double measure() const { return measure_; }
    // the smallest rectangle containing the simplex
    std::pair<SVector<embed_dim>, SVector<embed_dim>> bounding_box() const {
        return std::make_pair(coords_.rowwise().minCoeff(), coords_.rowwise().maxCoeff());
    }
    // the barycenter has all its barycentric coordinates equal to 1/(local_dim + 1)
    SVector<embed_dim> barycenter() const {
        return J_ * SVector<local_dim>::Constant(1.0 / (local_dim + 1)) + coords_.col(0);
    }

    // simplex circumcenter
    SVector<embed_dim> circumcenter() const {
        if constexpr (local_dim == 1) { return (coords_.col(0) + coords_.col(1)) / 2; }
        if constexpr (local_dim == 2 && embed_dim == 3) {
            // circumcenter of 3D triangle, see https://ics.uci.edu/~eppstein/junkyard/circumcenter.html
            SVector<embed_dim> a = coords_.col(1) - coords_.col(0);
            SVector<embed_dim> b = coords_.col(2) - coords_.col(0);
            SVector<embed_dim> aXb = a.cross(b);
            return coords_.col(0) +
                   (aXb.cross(a) * b.squaredNorm() + b.cross(aXb) * a.squaredNorm()) / (2 * aXb.squaredNorm());
        }
        if constexpr (local_dim == embed_dim && local_dim != 1) {
            // circumcenter of d-dimensional simplex
            // see Bruno Lévy, Yang Liu. Lp Centroidal Voronoi Tesselation and its applications. ACM(2010), Appendix B.2
            double a = coords_.col(0).squaredNorm();
            SMatrix<embed_dim, embed_dim> M;
            SVector<embed_dim> b;
            for (int i = 0; i < n_vertices - 1; ++i) {
                M.row(i) = coords_.col(i + 1) - coords_.col(0);
                b[i] = coords_.col(i + 1).squaredNorm() - a;
            }
            return 0.5 * M.inverse() * b;
        }
    }
    // simplex's circumcircle radius
    double circumradius() const { return (circumcenter() - coords_.col(0)).norm(); }

    // access to the i-th face as an Order_ - 1 Simplex
    Simplex<Order_ - 1, EmbedDim_> face(int i) const requires(Order_ > 0) {
        fdapde_assert(i < n_faces);
        std::vector<bool> bitmask(n_vertices, 0);
        std::fill_n(bitmask.begin(), n_vertices_per_face, 1);
        SMatrix<embed_dim, n_vertices_per_face> coords;
        for (int j = 0; j < i; ++j) std::prev_permutation(bitmask.begin(), bitmask.end());
        for (int j = 0, h = 0; j < n_vertices; ++j) {
            if (bitmask[j]) coords.col(h++) = coords_.col(j);
        }
        return Simplex<Order_ - 1, EmbedDim_>(coords);
    }
    // (hyper)plane passing thorught the simplex
    HyperPlane<local_dim, embed_dim> supporting_plane() const requires(local_dim != embed_dim) {
        if (!plane_.has_value()) { plane_ = HyperPlane<local_dim, embed_dim>(coords_); }
        return plane_.value();
    }
    // normal direction
    SVector<local_dim + 1> normal() const requires(local_dim != embed_dim) {
        return supporting_plane().normal();
    }
    // returns true if x belongs to the interior of the simplex
    enum ContainsReturnType { OUTSIDE = 0, INSIDE = 1, ON_FACE = 2, ON_VERTEX = 3 };
    ContainsReturnType contains(const SVector<embed_dim>& x) const requires(Order_ > 0) {
        if constexpr (local_dim != embed_dim) {
            if (supporting_plane().distance(x) > fdapde::machine_epsilon) return ContainsReturnType::OUTSIDE;
        }
        // move x to barycentric coordinates
        SVector<local_dim + 1> z;
        z.bottomRows(local_dim) = invJ_ * (x - coords_.col(0));
        z[0] = 1 - z.bottomRows(local_dim).sum();
        if ((z.array() < -fdapde::machine_epsilon).any()) return ContainsReturnType::OUTSIDE;
        int nonzeros = (z.array() > fdapde::machine_epsilon).count();
        if (nonzeros == 1) return ContainsReturnType::ON_VERTEX;
        if (nonzeros == n_vertices_per_face) return ContainsReturnType::ON_FACE;
        return ContainsReturnType::INSIDE;
    }
  
    // LegacyInputIterator over faces
    struct face_iterator {
       private:
        int index_;
        const Simplex* s_;
        FaceType f_;
       public:
        using value_type        = FaceType;
        using pointer           = const FaceType*;
        using reference         = const FaceType&;
        using size_type         = std::size_t;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        face_iterator(int index, const Simplex* s) : index_(index), s_(s) {
            if (index_ < s_->n_faces) f_ = s->face(index_);
        }
        reference operator*() const { return f_; }
        pointer operator->() const { return &f_; }
        face_iterator& operator++() {
            index_++;
            if (index_ < s_->n_faces) f_ = s_->face(index_);
            return *this;
        }
        face_iterator operator++(int) {
            face_iterator tmp(index_, this);
            ++(*this);
            return tmp;
        }
        friend bool operator!=(const face_iterator& lhs, const face_iterator& rhs) { return lhs.index_ != rhs.index_; }
        friend bool operator==(const face_iterator& lhs, const face_iterator& rhs) { return lhs.index_ == rhs.index_; }
    };
    face_iterator face_begin() const { return face_iterator(0, this); }
    face_iterator face_end() const { return face_iterator(n_faces, this); }
   protected:
    void initialize() {
        for (int j = 0; j < local_dim; ++j) { J_.col(j) = coords_.col(j + 1) - coords_.col(0); }
        if constexpr (embed_dim == local_dim) {
            invJ_ = J_.inverse();
            measure_ = std::abs(J_.determinant()) / (ct_factorial(local_dim));
        } else {   // generalized Penrose inverse for manifolds
            invJ_ = (J_.transpose() * J_).inverse() * J_.transpose();
            if constexpr (local_dim == 2) measure_ = 0.5 * J_.col(0).cross(J_.col(1)).norm();
            if constexpr (local_dim == 1) measure_ = J_.col(0).norm();
            if constexpr (local_dim == 0) measure_ = 0;   // points have zero measure
        }
    }

    SMatrix<embed_dim, n_vertices> coords_;
    mutable std::optional<HyperPlane<local_dim, embed_dim>> plane_;
    double measure_;
    // affine mappings from physical to reference simplex and viceversa
    SMatrix<embed_dim, local_dim> J_;      // [J_]_ij = (coords_(j,i) - coords_(0,i))
    SMatrix<local_dim, embed_dim> invJ_;   // J^{-1} (Penrose pseudo-inverse for manifold)
};

}   // namespace core
}   // namespace fdapde

#endif   // __SIMPLEX_H__
