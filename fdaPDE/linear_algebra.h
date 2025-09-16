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

#ifndef __FDAPDE_LINEAR_ALGEBRA_MODULE_H__
#define __FDAPDE_LINEAR_ALGEBRA_MODULE_H__

// clang-format off

// include Eigen linear algebra library
#include <Eigen/Eigen>
#define __FDAPDE_HAS_EIGEN__

namespace fdapde {
namespace internals {

// define basic eigen traits
template <typename XprType> struct is_eigen_dense_xpr {
    static constexpr bool value =
        std::is_base_of<Eigen::MatrixBase<std::decay_t<XprType>>, std::decay_t<XprType>>::value;
};
template <typename XprType> constexpr bool is_eigen_dense_xpr_v = is_eigen_dense_xpr<XprType>::value;
template <typename XprType> class is_eigen_dense_vec {
   private:
    using XprType_ = std::decay_t<XprType>;
    static constexpr bool check_() {
        if constexpr (is_eigen_dense_xpr_v<XprType_>) {
	    return XprType_::IsVectorAtCompileTime;
        }
        return false;
    }
   public:
    static constexpr bool value = check_();
};
template <typename XprType> constexpr bool is_eigen_dense_vec_v = is_eigen_dense_vec<XprType>::value;

template <typename XprType> struct is_eigen_sparse_xpr {
    static constexpr bool value =
        std::is_base_of_v<Eigen::SparseMatrixBase<std::decay_t<XprType>>, std::decay_t<XprType>>;
};
template <typename XprType> constexpr bool is_eigen_sparse_xpr_v = is_eigen_sparse_xpr<XprType>::value;

}   // namespace internals
}   // namespace fdapde

// include required modules
#include "utility.h"


namespace fdapde {

// forward declaration
template <int Rows, int Cols, typename XprType> struct MatrixExpr;

// storage orders
[[maybe_unused]] constexpr int RowMajor = 0;
[[maybe_unused]] constexpr int ColMajor = 1;
// triangular views
[[maybe_unused]] constexpr int Upper = 0;       // lower triangular view of matrix
[[maybe_unused]] constexpr int Lower = 1;       // upper triangular view of matrix
[[maybe_unused]] constexpr int UnitUpper = 2;   // lower triangular view of matrix with ones on the diagonal
[[maybe_unused]] constexpr int UnitLower = 3;   // upper triangular view of matrix with ones on the diagonal

[[maybe_unused]] static constexpr int LhsMode = 0;
[[maybe_unused]] static constexpr int RhsMode = 1;
  
namespace internals {

// detects whether XprType represents a static sized or dynamic sized expression
template <typename XprType> struct is_dynamic_sized {
   private:
    using XprTypeClean = std::decay_t<XprType>;
   public:
    static constexpr bool value = XprTypeClean::Rows == Dynamic || XprTypeClean::Cols == Dynamic;
};
template <typename XprType> static constexpr bool is_dynamic_sized_v = is_dynamic_sized<XprType>::value;

// if XprType has its NestAsRef bit set, sets type member type to XprType&, otherwise just repeats XprType
template <typename XprType, bool has_ref_bit> struct ref_select_impl;
template <typename XprType> struct ref_select_impl<XprType, true> {
   private:
    using XprTypeClean = std::decay_t<XprType>;
   public:
    using type = std::conditional_t<
      XprTypeClean::NestAsRef == 0, std::remove_reference_t<XprType>, std::add_lvalue_reference_t<XprType>>;
};
template <typename XprType> struct ref_select_impl<XprType, false> : std::type_identity<XprType> { };
template <typename XprType> struct ref_select {
    using type = ref_select_impl<XprType, requires(XprType) { XprType::NestAsRef; }>::type;
};
template <typename XprType> using ref_select_t = typename ref_select<XprType>::type;

}   // namespace internals
}

#include "src/linear_algebra/matrix.h"
#include "src/linear_algebra/diagonal.h"
#include "src/linear_algebra/triangular.h"

// special matrices
#include "src/linear_algebra/orthogonal.h"
#include "src/linear_algebra/symmetric.h"
#include "src/linear_algebra/skew.h"
#include "src/linear_algebra/permutation.h"
#include "src/linear_algebra/spd.h"

#include "src/linear_algebra/matrix_expr.h"

// algorithms
#include "src/linear_algebra/evd.h"
#include "src/linear_algebra/partial_piv_lu.h"
#include "src/linear_algebra/qr.h"

// eigen support
#include "src/linear_algebra/eigen/utility.h"

#include "src/linear_algebra/eigen/eigen_helper.h"
#include "src/linear_algebra/eigen/fspai.h"
#include "src/linear_algebra/eigen/kronecker.h"
#include "src/linear_algebra/eigen/lumping.h"
#include "src/linear_algebra/eigen/sparse_block_matrix.h"
#include "src/linear_algebra/eigen/woodbury.h"
// randomized linear algebra
#include "src/linear_algebra/eigen/rsi.h"
#include "src/linear_algebra/eigen/rbki.h"
#include "src/linear_algebra/eigen/rp_chol.h"

// clang-format on

#endif   // __FDAPDE_LINEAR_ALGEBRA_MODULE_H__
