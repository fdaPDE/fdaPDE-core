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

// a template class to perform geometric operations in general vector and affine spaces.
// M is the vector space dimension, N is the dimension of the embedding space
template <int M, int N> class VectorSpace {
   private:
    std::array<SVector<N>, M> basis_ {};   // the set of vectors generating the vector space
    SVector<N> offset_ {};   // a point throught which the vector space passes. (offset_ \neq 0 \implies affine space)

    // orthogonal projection of vector v over the space spanned by u
    SVector<N> orthogonal_project(const SVector<N>& u, const SVector<N>& v) const;
    void orthonormalize();   // basis orthonormalization via modified Gram-Schmidt method
   public:
    // constructor
    VectorSpace() = default;
    VectorSpace(const std::array<SVector<N>, M>& basis, const SVector<N>& offset) : basis_(basis), offset_(offset) {
        orthonormalize();
    };
    VectorSpace(const std::array<SVector<N>, M>& basis) : VectorSpace(basis, SVector<N>::Zero()) {};

    SVector<M> project_onto(const SVector<N>& x);   // projects the point x onto
    SVector<N> project_into(const SVector<N>& x);   // projects the point x into
    double distance(const SVector<N>& x);           // euclidean distance between point x and this space
    SVector<N> operator()(const std::array<double, M>& coeffs) const;   // expansion of coeffs wrt basis_
};

// implementation details

template <int M, int N>
SVector<N> VectorSpace<M, N>::orthogonal_project(const SVector<N>& u, const SVector<N>& v) const {
    return ((v.dot(u) / u.squaredNorm()) * (u.array())).matrix();
}

template <int M, int N> void VectorSpace<M, N>::orthonormalize() {
    std::array<SVector<N>, M> orthonormal_basis;
    orthonormal_basis[0] = basis_[0] / basis_[0].norm();
    // implementation of the modified Gram-Schmidt method, see theory for details
    for (int i = 1; i < basis_.size(); ++i) {
        orthonormal_basis[i] = basis_[i];
        for (int j = 0; j < i; ++j) { orthonormal_basis[i] -= orthogonal_project(orthonormal_basis[j], basis_[i]); }
        orthonormal_basis[i] /= orthonormal_basis[i].norm();
    }
    basis_ = orthonormal_basis;
    return;
}

// projects the point x onto the space.
template <int M, int N> SVector<M> VectorSpace<M, N>::project_onto(const SVector<N>& x) {
    if constexpr (M == N) {   // you are projecting over the same space!
        return x;
    } else {
        // build the projection onto the space spanned by basis_
        SVector<M> projection;
        for (size_t i = 0; i < basis_.size(); ++i) { projection[i] = (x - offset_).dot(basis_[i]) / basis_[i].norm(); }
        return projection;
    }
}

// project the point x into the space
template <int M, int N> SVector<N> VectorSpace<M, N>::project_into(const SVector<N>& x) {
    // build the projection operator on the space spanned by the basis
    Eigen::Matrix<double, N, Eigen::Dynamic> A;
    A.resize(N, basis_.size());
    for (size_t i = 0; i < basis_.size(); ++i) { A.col(i) = basis_[i]; }
    // given the projection operator A*A^T, the projection of x is computed as
    // (A*A^T)*(x-offset_) + offset_ (works also for affine spaces)
    return (A * A.transpose()) * (x - offset_) + offset_;
}

// euclidean distance from x
template <int M, int N> double VectorSpace<M, N>::distance(const SVector<N>& x) {
    return (x - project_into(x)).squaredNorm();
}

// develops the linear combination of basis vectors with respect to the given vector of coefficients.
template <int M, int N>
SVector<N> VectorSpace<M, N>::operator()(const std::array<double, M>& coeffs) const {
    SVector<N> result = offset_;
    for (std::size_t i = 0; i < M; ++i) result += coeffs[i] * basis_[i];
    return result;
}

}   // namespace core
}   // namespace fdapde

#endif   // __VECTOR_SPACE_H__
