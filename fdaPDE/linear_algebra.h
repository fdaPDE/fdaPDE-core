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

#include "utility.h"
#include <iomanip>
#include <sstream>

namespace fdapde {

// forward declaration
template <typename XprType> struct MatrixExpr;
template <typename XprType> struct MatrixCoeffWiseExpr;
template <typename T1, typename T2> using promote_type_t = std::common_type_t<T1, T2>;

// storage orders
[[maybe_unused]] constexpr int RowMajor = 0;
[[maybe_unused]] constexpr int ColMajor = 1;
// triangular views
[[maybe_unused]] constexpr int Upper = 0;       // upper triangular view of matrix
[[maybe_unused]] constexpr int Lower = 1;       // lower triangular view of matrix

[[maybe_unused]] static constexpr int LhsMode = 0;
[[maybe_unused]] static constexpr int RhsMode = 1;

namespace internals {

// tag types to enable/disable costly data integrity checks
/// @brief requests validation of structured input
struct checked_t { };
/// @brief skips optional structured-input validation
struct unchecked_t { };

}   // namespace internals

[[maybe_unused]] inline constexpr internals::checked_t   checked   {};
[[maybe_unused]] inline constexpr internals::unchecked_t unchecked {};

}   // namespace fdapde

#include "src/linear_algebra/traits.h"
#include "src/linear_algebra/numeric.h"
#include "src/linear_algebra/matrix.h"
#include "src/linear_algebra/binary_op.h"
#include "src/linear_algebra/block.h"
#include "src/linear_algebra/unary_op.h"
#include "src/linear_algebra/ternary_op.h"
#include "src/linear_algebra/vectorwise.h"
#include "src/linear_algebra/diagonal.h"
#include "src/linear_algebra/orthogonal.h"
#include "src/linear_algebra/triangular.h"
#include "src/linear_algebra/bool.h"
#include "src/linear_algebra/mdarray.h"
#include "src/linear_algebra/coeffwise.h"
#include "src/linear_algebra/symmetric.h"
#include "src/linear_algebra/skew.h"
#include "src/linear_algebra/permutation.h"
#include "src/linear_algebra/partial_piv_lu.h"
#include "src/linear_algebra/qr.h"
#include "src/linear_algebra/evd.h"
#include "src/linear_algebra/xpr.h"

#include "src/linear_algebra/utility.h"

#include "src/linear_algebra/eigen_helper.h"
#include "src/linear_algebra/fspai.h"
#include "src/linear_algebra/kronecker.h"
#include "src/linear_algebra/lumping.h"
#include "src/linear_algebra/sparse_block_matrix.h"
#include "src/linear_algebra/woodbury.h"
// randomized linear algebra
#include "src/linear_algebra/rsi.h"
#include "src/linear_algebra/rbki.h"
#include "src/linear_algebra/rp_chol.h"

// clang-format on

#endif   // __FDAPDE_LINEAR_ALGEBRA_MODULE_H__
