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

#include <fdaPDE/linear_algebra.h>

#include <type_traits>

using legacy_matrix = fdapde::Matrix<double, 2, 2>;
using legacy_permutation = fdapde::PermutationMatrix<2>;
using legacy_lu = fdapde::PartialPivLU<legacy_matrix>;
using native_matrix = fdapde::linalg::Matrix<double, 2, 2>;
using native_diagonal = fdapde::linalg::DiagonalMatrix<double, 2>;
using native_lower_triangular = fdapde::linalg::LowerTriangularMatrix<double, 2, 2>;
using native_symmetric = fdapde::linalg::SymmetricMatrix<double, 2, 2>;
using native_skew_symmetric = fdapde::linalg::SkewSymmetricMatrix<double, 2, 2>;
using native_orthogonal = fdapde::linalg::OrthogonalMatrix<double, 2, 2>;
using native_permutation = fdapde::linalg::PermutationMatrix<2, 2>;
using native_lu = fdapde::linalg::PartialPivLU<native_matrix>;
using native_qr = fdapde::linalg::HouseholderQR<double, 2, 2>;
using native_evd = fdapde::linalg::EVD<native_symmetric>;
using native_preconditioner = fdapde::linalg::IdentityPreconditioner<native_matrix>;
using native_gmres = fdapde::linalg::GMRES<native_matrix, native_preconditioner>;

static_assert(!std::is_same_v<legacy_matrix, native_matrix>);
static_assert(!std::is_same_v<legacy_permutation, native_permutation>);
static_assert(!std::is_same_v<legacy_lu, native_lu>);
static_assert(native_matrix::StorageOrder == fdapde::linalg::RowMajor);
static_assert(fdapde::Upper == fdapde::linalg::Upper);
static_assert(fdapde::Lower == fdapde::linalg::Lower);
static_assert(fdapde::linalg::is_diagonal_matrix_v<native_diagonal>);
static_assert(fdapde::linalg::is_triangular_matrix_v<native_lower_triangular>);
static_assert(fdapde::linalg::is_symmetric_matrix_v<native_symmetric>);
static_assert(fdapde::linalg::is_skew_symmetric_matrix_v<native_skew_symmetric>);
static_assert(fdapde::linalg::is_orthogonal_matrix_v<native_orthogonal>);
static_assert(fdapde::linalg::is_permutation_matrix_v<native_permutation>);
static_assert(std::is_default_constructible_v<native_lu>);
static_assert(std::is_default_constructible_v<native_qr>);
static_assert(std::is_default_constructible_v<native_evd>);
static_assert(std::is_default_constructible_v<native_preconditioner>);
static_assert(!std::is_default_constructible_v<native_gmres>);

[[maybe_unused]] legacy_matrix legacy_header_probe;
[[maybe_unused]] native_matrix native_header_probe;
