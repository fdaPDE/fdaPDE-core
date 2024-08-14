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
#include <numeric>

#include "../linear_algebra/binary_matrix.h"
#include "../utils/symbols.h"
#include "hyperplane.h"
#include "utils.h"

namespace fdapde {

// The convex-hull of EmbedDim_ + 1 points in \mathbb{R}^EmbedDim.
// A point (Order 0), line (Order 1), triangle (Order 2), tetrahedron (Order 3) embedded in \mathbb{R}^EmbedDim
template <int Order_, int EmbedDim_> class Simplex {
    static_assert(EmbedDim_ != 0 && Order_ <= 3);
   public:
    static constexpr int local_dim = Order_;
    static constexpr int embed_dim = EmbedDim_;
    static constexpr int n_nodes = Order_ + 1;
    static constexpr int n_edges = (Order_ * (Order_ + 1)) / 2;
    static constexpr int n_faces = Order_ + 1;
    static constexpr int n_nodes_per_face = Order_;
    using BoundaryCellType = Simplex<Order_ - 1, EmbedDim_>;
    using NodeType = SVector<embed_dim>;

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
    NodeType node(int v) const { return coords_.col(v); }
    NodeType operator[](int v) const { return coords_.col(v); }
    const SMatrix<embed_dim, n_nodes>& nodes() const { return coords_; }
    const SMatrix<embed_dim, local_dim>& J() const { return J_; }
    const SMatrix<local_dim, embed_dim>& invJ() const { return invJ_; }
    double measure() const { return measure_; }
    // the smallest rectangle containing the simplex
    std::pair<NodeType, NodeType> bounding_box() const {
        return std::make_pair(coords_.rowwise().minCoeff(), coords_.rowwise().maxCoeff());
    }
    // the barycenter has all its barycentric coordinates equal to 1/(local_dim + 1)
    NodeType barycenter() const {
        return J_ * SVector<local_dim>::Constant(1.0 / (local_dim + 1)) + coords_.col(0);
    }
    // writes the point p in the barycentric coordinate system of this simplex
    SVector<local_dim + 1> barycentric_coords(const NodeType& p) const {
        SVector<local_dim + 1> z;
        z.bottomRows(local_dim) = invJ_ * (p - coords_.col(0));
        z[0] = 1 - z.bottomRows(local_dim).sum();
        return z;
    }

    // simplex circumcenter
    NodeType circumcenter() const {
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
            for (int i = 0; i < n_nodes - 1; ++i) {
                M.row(i) = coords_.col(i + 1) - coords_.col(0);
                b[i] = coords_.col(i + 1).squaredNorm() - a;
            }
            return 0.5 * M.inverse() * b;
        }
    }
    // simplex's circumcircle radius
    double circumradius() const { return (circumcenter() - coords_.col(0)).norm(); }

    // simplex's diameter (length of the longest side)
    double diameter() const {
        double max_length = -1;
        for (int i = 0; i < n_nodes - 1; ++i) {
            SVector<embed_dim> c = coords_.col(i);
            for (int j = i + 1; j < n_nodes; ++j) {
                double length = (c - coords_.col(j)).norm();
                if (length > max_length) max_length = length;
            }
        }
        return max_length;
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
        if (nonzeros == n_nodes_per_face) return ContainsReturnType::ON_FACE;
        return ContainsReturnType::INSIDE;
    }

    // iterator over boundary faces
    class boundary_iterator : public index_based_iterator<boundary_iterator, BoundaryCellType> {
        using Base = index_based_iterator<boundary_iterator, BoundaryCellType>;
        using Base::index_;
        friend Base;
        const Simplex* s_;
        // access to the i-th boundary cell as an Order_ - 1 Simplex
        boundary_iterator& operator()(int i) requires(Order_ > 0) {
            std::vector<bool> bitmask(n_nodes, 0);
            std::fill_n(bitmask.begin(), n_nodes_per_face, 1);
            SMatrix<embed_dim, n_nodes_per_face> coords;
            for (int j = 0; j < i; ++j) std::prev_permutation(bitmask.begin(), bitmask.end());
            for (int j = 0, h = 0; j < n_nodes; ++j) {
                if (bitmask[j]) coords.col(h++) = s_->coords_.col(j);
            }
            Base::val_ = BoundaryCellType(coords);
            return *this;
        }
       public:
        boundary_iterator(int index, const Simplex* s) : Base(index, 0, Order_ + 1), s_(s) {
            if (index_ < Order_ + 1) operator()(index_);
        }
    };
    boundary_iterator boundary_begin() const requires(Order_ >= 1) { return boundary_iterator(0, this); }
    boundary_iterator boundary_end() const requires(Order_ >= 1) { return boundary_iterator(Order_ + 1, this); }

    // finds the best approximation of p in the simplex (q \in simplex : q = \argmin_{t \in simplex}{\norm{t - p}})
    SVector<embed_dim> nearest(const SVector<embed_dim>& p) const {
        SVector<local_dim + 1> q = barycentric_coords(p);
	// check if point inside simplex
        if constexpr (local_dim != embed_dim) {
            if (
              (q.array() > -fdapde::machine_epsilon).all() && supporting_plane().distance(p) < fdapde::machine_epsilon)
                return p;
        } else {
            if ((q.array() > -fdapde::machine_epsilon).all()) return p;
        }
        if constexpr (Order_ == 1) {   // end of recursion
            if (q[0] < 0) return coords_.col(1);
            return coords_.col(0);
        } else {
            // find nearest face
            std::array<double, n_nodes> dst;
            for (int i = 0; i < n_nodes; ++i) { dst[i] = (coords_.col(i) - p).norm(); }
            std::array<int, n_nodes> idx;
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return dst[a] < dst[b]; });
	    // recurse on Order_ - 1 subsimplex
            Simplex<Order_ - 1, embed_dim> s(coords_(Eigen::all, std::vector<int>(idx.begin(), idx.end() - 1)));
            return s.nearest(p);
        }
    }
  
   protected:
    void initialize() {
        for (int j = 0; j < n_nodes - 1; ++j) { J_.col(j) = coords_.col(j + 1) - coords_.col(0); }
        if constexpr (embed_dim == local_dim) {
            invJ_ = J_.inverse();
            measure_ = std::abs(J_.determinant()) / (cexpr::factorial(local_dim));
        } else {   // generalized Penrose inverse for manifolds
            invJ_ = (J_.transpose() * J_).inverse() * J_.transpose();
            if constexpr (local_dim == 2) measure_ = 0.5 * J_.col(0).cross(J_.col(1)).norm();
            if constexpr (local_dim == 1) measure_ = J_.col(0).norm();
            if constexpr (local_dim == 0) measure_ = 0;   // points have zero measure
        }
    }

    SMatrix<embed_dim, n_nodes> coords_;
    mutable std::optional<HyperPlane<local_dim, embed_dim>> plane_;
    double measure_;
    // affine mappings from physical to reference simplex and viceversa
    SMatrix<embed_dim, local_dim> J_;      // [J_]_ij = (coords_(j,i) - coords_(0,i))
    SMatrix<local_dim, embed_dim> invJ_;   // J^{-1} (Penrose pseudo-inverse for manifold)
};

}   // namespace fdapde

#endif   // __SIMPLEX_H__
