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

#ifndef __FDAPDE_HYPERPLANE_H__
#define __FDAPDE_HYPERPLANE_H__

#include "header_check.h"

namespace fdapde {

// a template class representing an M-dimensional plane embedded in an N-dimensional space
template <int LocalDim, int EmbedDim> class HyperPlane {
    static_assert(LocalDim <= EmbedDim);
   public:
    static constexpr int embed_dim = EmbedDim;
    static constexpr int local_dim = LocalDim;
    HyperPlane() = default;
    // constructs a hyperplane passing through 2 points, e.g. a line
    HyperPlane(const Eigen::Matrix<double, embed_dim, 1>& x1, const Eigen::Matrix<double, embed_dim, 1>& x2) : p_(x1) {
        fdapde_static_assert(local_dim == 1, THIS_METHOD_IS_ONLY_FOR_LINES);
	Eigen::Matrix<double, embed_dim, 1> tmp = x2 - x1;
        basis_.col(0) = tmp.normalized();
        if constexpr (embed_dim == 2) normal_ << -tmp[0], tmp[1];
        if constexpr (embed_dim == 3) normal_ << -tmp[0], tmp[1], 0;   // one of the (infintely-many) normals to 3D line
        normal_ = normal_.normalized();
        offset_ = -x1.dot(normal_);
    }
    // constructs an hyperplane passing through 3 (non-collinear) points, e.g. a plane
    HyperPlane(
      const Eigen::Matrix<double, embed_dim, 1>& x1, const Eigen::Matrix<double, embed_dim, 1>& x2,
      const Eigen::Matrix<double, embed_dim, 1>& x3) :
        p_(x1) {
        fdapde_static_assert(local_dim == 2, THIS_METHOD_IS_ONLY_FOR_PLANES);
        basis_.col(0) = (x2 - x1).normalized();
        basis_.col(1) = ((x3 - x1) - orthogonal_project(
                                       Eigen::Matrix<double, embed_dim, 1>(x3 - x1),
                                       Eigen::Matrix<double, embed_dim, 1>(basis_.col(0))))
                          .normalized();
        normal_ = ((x2 - x1).cross(x3 - x1)).normalized();
        offset_ = -x1.dot(normal_);
    }
    // constructors from matrix coordinates
    HyperPlane(const Eigen::Matrix<double, embed_dim, 2>& coords) requires(local_dim == 1) :
      HyperPlane(coords.col(0), coords.col(1)) { }
    HyperPlane(const Eigen::Matrix<double, embed_dim, 3>& coords) requires (local_dim == 2) :
      HyperPlane(coords.col(0), coords.col(1), coords.col(2)) { }
    // general hyperplane constructor
    HyperPlane(const Eigen::Matrix<double, embed_dim, local_dim + 1>& coords) requires(local_dim > 2) :
        p_(coords.col(0)) {
        basis_ = coords.rightCols(local_dim).colwise() - coords.col(0);
	// basis orthonormalization via modified Gram-Schmidt method
        basis_.col(0) /= basis_.col(0).norm();
        for (int i = 1; i < local_dim; ++i) {
            for (int j = 0; j < i; ++j) {
                basis_.col(i) = basis_.col(i) - orthogonal_project(basis_.col(i), basis_.col(j));
            }
            basis_.col(i) /= basis_.col(i).norm();
        }
        normal_ = basis_.fullPivLu().kernel();   // normal to the hyperplane is any element in the null space of basis_
        offset_ = -coords.col(0).dot(normal_);
    }
    // projection
    Eigen::Matrix<double, local_dim, 1> project_onto(const Eigen::Matrix<double, embed_dim, 1>& x) {
        if constexpr (local_dim == embed_dim) {
            return x;
        } else {
            // build the projection onto the space spanned by basis_
            Eigen::Matrix<double, local_dim, 1> proj;
            for (int i = 0; i < local_dim; ++i) { proj[i] = (x - p_).dot(basis_.col(i)); }
            return proj;
        }
    }
    Eigen::Matrix<double, embed_dim, 1> project(const Eigen::Matrix<double, embed_dim, 1>& x) const {
        if constexpr (local_dim == embed_dim) {
            return x;
        } else {
            return basis_ * basis_.transpose() * (x - p_) + p_;
        }
    }
    double distance(const Eigen::Matrix<double, embed_dim, 1>& x) {
        return (x - project(x)).norm();
    }   // point-plane distance
    double eval(const Eigen::Matrix<double, embed_dim, 1>& coeffs) const { return normal_.dot(coeffs) + offset_; }
    Eigen::Matrix<double, embed_dim, 1> operator()(const Eigen::Matrix<double, local_dim, 1>& coeffs) const {
        Eigen::Matrix<double, embed_dim, 1> res = p_;
        for (int i = 0; i < local_dim; ++i) { res += coeffs[i] * basis_.col(i); }
        return res;
    }
    const Eigen::Matrix<double, embed_dim, 1>& normal() const { return normal_; }   // plane normal direction
    const Eigen::Matrix<double, embed_dim, 1>& point() const { return p_; }         // a point belonging to the plane
    const Eigen::Matrix<double, embed_dim, local_dim>& basis() const { return basis_; }
   private:
    // let x_1, x_2, \ldots, x_{N+1} be a set of N+1 points through which the plane passes
    Eigen::Matrix<double, embed_dim, local_dim> basis_;   // matrix [x2 - x1, x3 - x1, \ldots, x_{N+1} - x1]
    Eigen::Matrix<double, embed_dim, 1> normal_;         // normal vector to the hyperplane
    Eigen::Matrix<double, embed_dim, 1> p_;              // point through which the plane is guaranteed to pass
    double offset_;   // hyperplane's intercept (the d in the equation ax + by + cz + d = 0)

    // orthogonal projection of vector v over u
    template <int N>
    constexpr Eigen::Matrix<double, N, 1>
    orthogonal_project(const Eigen::Matrix<double, N, 1>& v, const Eigen::Matrix<double, N, 1>& u) {
        return (v.dot(u) / u.squaredNorm() * u.array()).matrix();
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_HYPERPLANE_H__
