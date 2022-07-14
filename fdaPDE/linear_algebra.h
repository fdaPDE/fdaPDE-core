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
#include "src/linear_algebra/utility.h"

#include "linear_algebra/eigen_helper.h"
#include "linear_algebra/fspai.h"
#include "linear_algebra/kronecker.h"
#include "linear_algebra/lumping.h"
#include "linear_algebra/sparse_block_matrix.h"
#include "linear_algebra/fspai.h"
#include "linear_algebra/lumping.h"

#include "fdaPDE/linear_algebra/randomized_algorithms/rand_range_finder.h"
#include "fdaPDE/linear_algebra/randomized_algorithms/randomized_svd.h"
#include "fdaPDE/linear_algebra/randomized_algorithms/rand_nys_approximation.h"


#include "fdaPDE/linear_algebra/randomized_algorithms/rsvd.h"
#include "fdaPDE/linear_algebra/randomized_algorithms/revd.h"
#include "fdaPDE/linear_algebra/randomized_algorithms/nystrom.h"

// clang-format on

#endif   // __FDAPDE_LINEAR_ALGEBRA_MODULE_H__