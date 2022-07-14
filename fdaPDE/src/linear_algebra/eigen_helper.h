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

#ifndef __FDAPDE_EIGEN_HELPER_H__
#define __FDAPDE_EIGEN_HELPER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {
  
template <int Rows, typename Scalar = double> struct static_dynamic_eigen_vector_selector {
    using type = std::conditional_t<Rows == Dynamic, Eigen::Matrix<Scalar, Dynamic, 1>, Eigen::Matrix<Scalar, Rows, 1>>;
};
template <int Rows, typename Scalar = double>
using static_dynamic_eigen_vector_selector_t = typename static_dynamic_eigen_vector_selector<Rows, Scalar>::type;

template <int Rows, int Cols, typename Scalar = double> struct static_dynamic_eigen_matrix_selector {
    using type = typename switch_type<
      switch_type_case<Rows == Dynamic && Cols == Dynamic, Eigen::Matrix<Scalar, Dynamic, Dynamic>>,
      switch_type_case<Rows == Dynamic && Cols != Dynamic, Eigen::Matrix<Scalar, Rows, Dynamic>>,
      switch_type_case<Rows != Dynamic && Cols == Dynamic, Eigen::Matrix<Scalar, Dynamic, Cols>>,
      switch_type_case<Rows != Dynamic && Cols != Dynamic, Eigen::Matrix<Scalar, Rows, Cols>>>::type;
};
template <int Rows, int Cols, typename Scalar = double>
using static_dynamic_matrix_selector_t = typename static_dynamic_eigen_matrix_selector<Rows, Cols, Scalar>::type;

// a movable wrapper for Eigen::SparseLU (Eigen::SparseLU has a deleted copy and assignment operator)
template <typename SolverType_> class eigen_sparse_solver_movable_wrap {
   private:
    using SolverType = std::decay_t<SolverType_>;
    std::shared_ptr<SolverType> solver_;   // wrap solver in movable wrapper
    bool computed_ = false;
   public:
    using Scalar = typename SolverType::Scalar;
    using MatrixType = typename SolverType::MatrixType;

    eigen_sparse_solver_movable_wrap() noexcept : solver_(std::make_shared<SolverType>()) { }
    explicit eigen_sparse_solver_movable_wrap(const SolverType_& solver) noexcept :
        solver_(std::make_shared<SolverType>(solver)) { }
    eigen_sparse_solver_movable_wrap& operator=(const SolverType_& solver) {
        solver_ = std::make_shared<SolverType>(solver);
        return *this;
    }
    void compute(const MatrixType& matrix) {
        solver_->compute(matrix);
        if (solver_->info() == Eigen::Success) { computed_ = true; }
    }
    void analyzePattern(const MatrixType& matrix) { solver_->analyzePattern(matrix); }
    void factorize(const MatrixType& matrix) { solver_->factorize(matrix); }
    template <typename XprType>   // solve method, dense  rhs operand
    const Eigen::Solve<SolverType, XprType> solve(const Eigen::MatrixBase<XprType>& b) const {
        fdapde_assert(bool(solver_) == true);
        return solver_->solve(b);
    }
    template <typename XprType>   // solve method, sparse rhs operand
    const Eigen::Solve<SolverType, XprType> solve(const Eigen::SparseMatrixBase<XprType>& b) const {
        fdapde_assert(bool(solver_) == true);
        return solver_->solve(b);
    }
    // observers
    const SolverType& operator->() const { return *solver_; }
    SolverType& operator->() { return *solver_; }
    Eigen::ComputationInfo info() const { return solver_->info(); }
    operator bool() const { return computed_; }
    bool has_value() const { return computed_; }
};
  
// ordering relation for eigen vectors
struct eigen_vector_compare {
  template <typename Scalar, int Rows>
    bool operator()(const Eigen::Matrix<Scalar, Rows, 1>& lhs, const Eigen::Matrix<Scalar, Rows, 1>& rhs) const {
        return std::lexicographical_compare(lhs.data(), lhs.data() + lhs.size(), rhs.data(), rhs.data() + rhs.size());
    }
};
// hash function for Eigen matrices
struct eigen_matrix_hash {
    template <typename Scalar, int Rows, int Cols>
    std::size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols>& matrix) const {
        std::size_t hash = 0;
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                hash ^= std::hash<Scalar>()(matrix(i, j)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
        }
        return hash;
    }
};

}   // namespace internals
  
// A Triplet type (almost identical to Eigen::Triplet<T>) but allowing for non-const access of stored values.
template <typename Scalar_> class Triplet {
   public:
    using Index = int;
    using Scalar = std::decay_t<Scalar_>;

    Triplet() : row_(0), col_(0), value_() { }
    Triplet(int row, int col, const Scalar_& value) : row_(row), col_(col), value_(value) { }
    int row() const { return row_; }
    int col() const { return col_; }
    const Scalar& value() const { return value_; }
    Scalar& value() { return value_; }
   private:
    Index row_, col_;
    Scalar value_;
};

// test if Eigen matrix is empty (a zero-sized matrix is considered empty)
template <typename Derived> inline bool is_empty(const Eigen::EigenBase<Derived>& matrix) { return matrix.size() == 0; }

}   // namespace fdapde

#endif   // __FDAPDE_EIGEN_HELPER_H__
