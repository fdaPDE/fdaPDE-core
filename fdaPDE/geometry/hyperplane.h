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

#ifndef __VECTOR_SPACE_H__
#define __VECTOR_SPACE_H__

#include <array>

#include "../utils/assert.h"
#include "../utils/symbols.h"

namespace fdapde {
namespace core {

// orthogonal projection of vector v over u
template <int N> constexpr SVector<N> orthogonal_project(const SVector<N>& v, const SVector<N>& u) {
    return (v.dot(u) / u.squaredNorm() * u.array()).matrix();
}

// a template class representing an M-dimensional plane embedded in an N-dimensional space
template <int M, int N> class HyperPlane {
    static_assert(M <= N);
   private:
    // let x_1, x_2, \ldots, x_{N+1} be a set of N+1 points through which the plane passes
    SMatrix<N, M> basis_;   // matrix [x2 - x1, x3 - x1, \ldots, x_{N+1} - x1] of vectors generating the plane
    double offset_;         // hyperplane's intercept (the d in the equation ax + by + cz + d = 0)
    SVector<N> normal_;     // normal vector to the hyperplane
    SVector<N> p_;          // point through which the plane is guaranteed to pass
   public:
    // constructor
    HyperPlane() = default;
    // constructs a hyperplane passing through 2 points, e.g. a line
    HyperPlane(const SVector<N>& x1, const SVector<N>& x2) : p_(x1) {
        fdapde_static_assert(M == 1, THIS_METHOD_IS_ONLY_FOR_LINES);
	SVector<N> tmp = x2 - x1;
        basis_.col(0) = tmp.normalized();
        if constexpr (N == 2) normal_ << -tmp[0], tmp[1];
        if constexpr (N == 3) normal_ << -tmp[0], tmp[1], 0;   // one of the (infintely-many) normals to a 3D line
        normal_.normalized();
        offset_ = -x1.dot(normal_);
    }
    // constructs an hyperplane passing through 3 (non-collinear) points, e.g. a plane
    HyperPlane(const SVector<N>& x1, const SVector<N>& x2, const SVector<N> x3) : p_(x1) {
        fdapde_static_assert(M == 2, THIS_METHOD_IS_ONLY_FOR_PLANES);
        basis_.col(0) = (x2 - x1).normalized();
        basis_.col(1) = ((x3 - x1) - orthogonal_project(SVector<N>(x3 - x1), SVector<N>(basis_.col(0)))).normalized();
	normal_ = ((x2 - x1).cross(x3 - x1)).normalized();
	offset_ = -x1.dot(normal_);
    }
    // constructors from matrix coordinates
    HyperPlane(const SMatrix<N, 2>& coords) requires (M == 1) : HyperPlane(coords.col(0), coords.col(1)) { }
    HyperPlane(const SMatrix<N, 3>& coords) requires (M == 2) :
      HyperPlane(coords.col(0), coords.col(1), coords.col(2)) { }
    // general hyperplane constructor
    HyperPlane(const SMatrix<N, M + 1>& coords) requires(M > 2) : p_(coords.col(0)) {
        basis_ = coords.rightCols(M).colwise() - coords.col(0);
	// basis orthonormalization via modified Gram-Schmidt method
        basis_.col(0) /= basis_.col(0).norm();
        for (int i = 1; i < M; ++i) {
            for (int j = 0; j < i; ++j) {
                basis_.col(i) = basis_.col(i) - orthogonal_project(basis_.col(i), basis_.col(j));
            }
            basis_.col(i) /= basis_.col(i).norm();
        }
        normal_ = basis_.fullPivLu().kernel();   // normal to the hyperplane is any element in the null space of basis_
        offset_ = -coords.col(0).dot(normal_);
    }
    // projection
    SVector<M> project_onto(const SVector<N>& x) {
        if constexpr (M == N) {
            return x;
        } else {
            // build the projection onto the space spanned by basis_
            SVector<M> proj;
            for (int i = 0; i < M; ++i) { proj[i] = (x - p_).dot(basis_.col(i)); }
            return proj;
        }
    }
    SVector<N> project(const SVector<N>& x) const {
        if constexpr (M == N) {
            return x;
        } else {
            return basis_ * basis_.transpose() * (x - p_) + p_;
        }
    }
    double distance(const SVector<N>& x) { return (x - project(x)).norm(); }   // point-plane euclidean distance
    double eval(const SVector<N>& coeffs) const { return normal_.dot(coeffs) + offset_; }
    SVector<N> operator()(const SVector<M>& coeffs) const {
        SVector<N> res = p_;
        for (int i = 0; i < M; ++i) res += coeffs[i] * basis_.col(i);
        return res;
    }
    const SVector<N>& normal() const { return normal_; }   // normal direction to the hyperplane
    const SVector<N>& point() const { return p_; }         // a point belonging to the plane
    const SMatrix<N, M>& basis() const { return basis_; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __VECTOR_SPACE_H__
