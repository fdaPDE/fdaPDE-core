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

#ifndef __FDAPDE_LINEAR_ALGEBRA_UTILITY_H__
#define __FDAPDE_LINEAR_ALGEBRA_UTILITY_H__

#include "header_check.h"

namespace fdapde {
namespace internals {
    
// builds gaussian matrix
inline Eigen::Matrix<double, Dynamic, Dynamic>
gaussian_matrix(std::size_t rows, std::size_t cols, double std = 1.0, int seed = fdapde::random_seed) {
    // set up random number generation
    int seed_ = (seed == fdapde::random_seed) ? std::random_device()() : seed;
    std::mt19937 rng(seed_);
    std::normal_distribution norm_distr(0.0, std);
    // build random matrix
    Eigen::Matrix<double, Dynamic, Dynamic> m(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; j++) { m(i, j) = norm_distr(rng); }
    }
    return m;
}

// apply functor f to a symmetric matrix M by computing f(M) = S * f(\Sigma) * S^\top, where:
// S eigenvector matrix, \Sigma eigenvalue matrix
template <typename Derived, typename Functor>
Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
symm_matrix_apply(const Eigen::MatrixBase<Derived>& m, Functor&& f) {
    using matrix_t = Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
    using vector_t = Eigen::Matrix<double, Derived::RowsAtCompileTime, 1>;

    Eigen::SelfAdjointEigenSolver<Derived> eigen(m);
    vector_t eig_val = eigen.eigenvalues();
    matrix_t eig_vec = eigen.eigenvectors();
    // logaritm of eigenvalues
    for (int i = 0; i < m.derived().rows(); ++i) { eig_val[i] = f(eig_val[i]); }
    return eig_vec * eig_val.asDiagonal() * eig_vec.transpose();
}

}   // namespace internals

// matrix logarithm
template <typename Derived> auto logm(const Eigen::MatrixBase<Derived>& m) {
    return internals::symm_matrix_apply(m, [](double x) { return std::log(x); });
}
// matrix exponential
template <typename Derived> auto expm(const Eigen::MatrixBase<Derived>& m) {
    return internals::symm_matrix_apply(m, [](double x) { return std::exp(x); });
}
// matrix power
template <typename Derived> auto powm(const Eigen::MatrixBase<Derived>& m, int i) {
    return internals::symm_matrix_apply(m, [&](double x) { return std::pow(x, i); });
}
template <typename Derived> auto sqrtm(const Eigen::MatrixBase<Derived>& m) { return powm(m, 0.5); }

}   // namespace fdapde

#endif // __FDAPDE_LINEAR_ALGEBRA_UTILITY__
