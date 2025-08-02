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

#ifndef __FDAPDE_MATRIX_H__
#define __FDAPDE_MATRIX_H__

#include "header_check.h"
#include "matrix_base.h"

namespace fdapde {

template <typename Scalar_, int Rows_, int Cols_, int NestAsRefBit_ = 1>
class Matrix : public MatrixBase<Rows_, Cols_, Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>> {
    fdapde_static_assert(Rows_ > 0 && Cols_ > 0, EMPTY_MATRIX_IS_ILL_FORMED);
   public:
    using Base = MatrixBase<Rows_, Cols_, Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>>;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = NestAsRefBit_;   // whether to store this node by ref or by copy in an expression
    static constexpr int ReadOnly = 0;

    constexpr Matrix() : data_() {};
    constexpr explicit Matrix(const std::array<Scalar, Rows * Cols>& data) : data_(data) { }
    constexpr explicit Matrix(const std::vector<Scalar>& data) : data_() {
        fdapde_constexpr_assert(data.size() == Rows * Cols);
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) { data_[i * Cols + j] = data[i * Cols + j]; }
        }
    }
    template <typename Derived>
    constexpr Matrix(const MatrixBase<Rows_, Cols_, Derived>& xpr) : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_TYPES_CONVERSION_BETWEEN_MATRICES);
        fdapde_static_assert(
          Derived::Rows == Rows && Derived::Cols == Cols,
          YOU_ARE_TRYING_TO_CONSTRUCT_A_MATRIX_WITH_ANOTHER_MATRIX_OF_DIFFERENT_SIZE);
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) { data_[i * Cols + j] = xpr.derived().operator()(i, j); }
        }
    }
    template <typename Callable>
    constexpr explicit Matrix(Callable callable)
        requires(std::is_invocable_v<Callable>)
        : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {
            callable})::result_type FDAPDE_COMMA std::array<Scalar FDAPDE_COMMA Rows * Cols>>,
          CALLABLE_DOES_NOT_RETURN_SOMETHING_CONVERTIBLE_TO_AN_ARRAY_OF_SCALAR);
        data_ = callable();
    }
    constexpr explicit Matrix(const Scalar_ (&data)[Rows * Cols]) : data_() {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { data_[i * Cols + j] = data[i * Cols + j]; }
        }
    }
    constexpr explicit Matrix(Scalar x) : data_() {   // 1D point constructor
        fdapde_static_assert(Rows * Cols == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_ONE_ELEMENT);
	data_[0] = x;
    }
    constexpr explicit Matrix(Scalar x, Scalar y) : data_() {   // 2D point constructor
        fdapde_static_assert(Rows * Cols == 2, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_TWO_ELEMENTS);
	data_ = {x, y};
    }
    constexpr explicit Matrix(Scalar x, Scalar y, Scalar z) : data_() {   // 3D point constructor
        fdapde_static_assert(Rows * Cols == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_THREE_ELEMENTS);
	data_ = {x, y, z};
    }
    // static constructors
    static constexpr Matrix<Scalar, Rows, Cols> Constant(Scalar c) {
        Matrix<Scalar, Rows, Cols> m;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { m(i, j) = c; }
        }
        return m;
    }
    static constexpr Matrix<Scalar, Rows, Cols> Zero() { return Constant(Scalar(0)); }
    static constexpr Matrix<Scalar, Rows, Cols> Ones() { return Constant(Scalar(1)); }
    static constexpr Matrix<Scalar, Rows, Cols> NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }
    // const access
    constexpr Scalar operator()(int i, int j) const { return data_[i * Cols + j]; }
    constexpr Scalar operator[](int i) const
        requires(Cols == 1 || Rows == 1) {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_[i];
    }
    // non-const access
    constexpr Scalar& operator()(int i, int j) { return data_[i * Cols + j]; }
    constexpr Scalar& operator[](int i)
        requires(Cols == 1 || Rows == 1) {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_[i];
    }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr const Scalar* data() const { return data_.data(); }
    Scalar* data() { return data_.data(); }
    // assignment operator
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>&
    operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_DIMENSIONS_OR_YOU_ARE_TRYING_TO_ASSIGN_A_RHS_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs.derived()(i, j); }
        }
        return *this;
    }
    // assignment from std::array
    constexpr Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>& operator=(const std::array<Scalar_, Rows_ * Cols_>& rhs) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs[i * Cols + j]; }
        }
        return *this;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // conversion from Eigen matrix
        template <typename Derived> Matrix(const Eigen::MatrixBase<Derived>& other) {
            constexpr int Rows__ = Derived::RowsAtCompileTime;
            constexpr int Cols__ = Derived::ColsAtCompileTime;
            fdapde_static_assert(
              Rows__ != Dynamic && Cols__ != Dynamic && Rows__ == Rows && Cols__ == Cols &&
                std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
              INVALID_CONVERSION_FROM_EIGEN_MATRIX_TO_FDAPDE_MATRIX);
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < Cols; ++j) { operator()(i, j) = other(i, j); }
            }
        }
        Eigen::Map<Eigen::Matrix<Scalar_, Rows_, Cols_>> as_eigen_map() {
            return Eigen::Map<Eigen::Matrix<Scalar_, Rows_, Cols_>>(data_.data());
        }
        // assignment from Eigen matrix
        template <typename Derived>
        Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>& operator=(const Eigen::MatrixBase<Derived>& rhs) {
            fdapde_static_assert(
              Derived::RowsAtCompileTime != Dynamic && Derived::ColsAtCompileTime != Dynamic &&
                std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
              CANNOT_ASSIGN_FROM_EIGEN_HEAP_ALLOCATED_MATRIX_OR_INVALID_SCALAR_TYPE);
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs.derived()(i, j); }
            }
            return *this;
        }
    #endif

    constexpr void setConstant(Scalar c) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = c; }
        }
	return;
    }
    constexpr void setZero() { setConstant(Scalar(0)); }
    constexpr void setOnes() { setConstant(Scalar(1)); }
   private:
    std::array<Scalar, Rows * Cols> data_;
};

// alias export for constexpr-enabled vectors
template <typename Scalar_, int Rows_> using Vector = Matrix<Scalar_, Rows_, 1>;

template <int Size_> struct PermutationMatrix : public MatrixBase<Size_, Size_, PermutationMatrix<Size_>> {
    using Base = MatrixBase<Size_, Size_, PermutationMatrix<Size_>>;
    using Scalar = int;
    using XprType = PermutationMatrix<Size_>;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr PermutationMatrix() = default;
    constexpr explicit PermutationMatrix(const std::array<int, Size_>& permutation) : permutation_(permutation) { }
    constexpr int rows() const { return Size_; }
    constexpr int cols() const { return Size_; }
    // left multiplication by permutation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    constexpr Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& rhs) const {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
        using Scalar = typename RhsType::Scalar;
        Matrix<Scalar, Rows, RhsCols> permuted;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < RhsCols; ++j) { permuted(i, j) = rhs.derived().operator()(permutation_[i], j); }
        }
        return permuted;
    }
    // right multiplication by permutation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    constexpr friend Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& lhs, const PermutationMatrix<Size_>& rhs) {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERANDS_DIMENSION_FOR_MATRIX_MATRIX_PRODUCT);
        using Scalar = typename RhsType::Scalar;
        Matrix<Scalar, Rows, RhsCols> permuted;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < RhsCols; ++j) { permuted(j, i) = lhs.derived().operator()(j, rhs.permutation()[i]); }
        }
        return permuted;
    }
    constexpr const std::array<int, Size_>& permutation() const { return permutation_; }
   private:
    std::array<int, Size_> permutation_;
};

template <typename Matrix, typename Rhs> constexpr auto backward_sub(const Matrix& A, const Rhs& b) {
    fdapde_static_assert(
      std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
    // check dimensions
    using Scalar = typename Matrix::Scalar;
    constexpr int rows = Matrix::Rows;
    Vector<Scalar, rows> res;
    int i = rows - 1;
    res[i] = b[i] / A(i, i);
    i--;
    for (; i >= 0; --i) {
        Scalar tmp = 0;
        for (int j = i + 1; j < rows; ++j) tmp += A(i, j) * res[j];
        res[i] = 1. / A(i, i) * (b[i] - tmp);
    }
    return res;
}

template <typename Matrix, typename Rhs> constexpr auto forward_sub(const Matrix& A, const Rhs& b) {
    fdapde_static_assert(
      std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
    // check dimensions
    using Scalar = typename Matrix::Scalar;
    constexpr int rows = Matrix::Rows;
    Vector<Scalar, rows> res;
    int i = 0;
    res[i] = b[i] / A(i, i);
    i++;
    for (; i < rows; ++i) {
        Scalar tmp = 0;
        for (int j = 0; j < i; ++j) tmp += A(i, j) * res[j];
        res[i] = 1. / A(i, i) * (b[i] - tmp);
    }
    return res;
}

// LU factorization of matrix with partial pivoting
template <typename MatrixType> class PartialPivLU {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, LU_FACTORIZATION_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    static constexpr int Size = MatrixType::Rows;
    using Scalar = typename MatrixType::Scalar;
    MatrixType m_;
    PermutationMatrix<Size> P_;
   public:
    constexpr PartialPivLU() : m_(), P_() {};
    template <typename XprType> constexpr PartialPivLU(const MatrixBase<Size, Size, XprType>& m) : m_() { compute(m); }

    // computes the LU factorization of matrix m with partial (row) pivoting
    template <typename XprType> constexpr void compute(const MatrixBase<Size, Size, XprType>& m) {
        m_ = m;
        std::array<int, Size> P;
        for (int i = 0; i < Size; ++i) { P[i] = i; }
        int pivot_index = 0;
        int h, k;
        for (int i = 0; i < Size - 1; ++i) {
            // find pivotal element
            Scalar pivot = -std::numeric_limits<Scalar>::infinity();
            for (int j = i; j < Size; ++j) {
                Scalar abs_ = fdapde::abs(m_(P[j], i));
                if (pivot < abs_) {
                    pivot = abs_;
                    pivot_index = j;
                }
            }
            // perform gaussian elimination step in place
            for (int j = i; j < Size; ++j) {
                if (P[j] != P[pivot_index]) {   // avoid to subtract row with itself
                    Scalar l = m_(P[j], i) / m_(P[pivot_index], i);
                    m_(P[j], i) = l;
                    for (int k = i + 1; k < Size; ++k) { m_(P[j], k) = m_(P[j], k) - l * m_(P[pivot_index], k); }
                }
            }
            // swap rows
            h = P[i], k = P[pivot_index];
            P[pivot_index] = h;
            P[i] = k;
        }
        P_ = PermutationMatrix<Size>(P);
	m_ = P_ * m_;
    }
    constexpr PermutationMatrix<Size> P() const { return P_; }
    // solve linear system Ax = b using A factorization PA = LU
    template <typename RhsType> constexpr Matrix<Scalar, Size, 1> solve(const RhsType& rhs) {
        fdapde_static_assert(
          std::is_same_v<Scalar FDAPDE_COMMA typename RhsType::Scalar>, INVALID_SCALAR_TYPE_FOR_RHS_OPERAND);
        fdapde_constexpr_assert(rhs.rows() == Size && rhs.cols() == 1);
        Matrix<Scalar, Size, 1> x;
        // evaluate U^{-1} * (L^{-1} * (P * rhs))
        x = P_ * rhs;
        x = forward_sub(m_.template triangular_view<UnitLower>(), x);
        x = backward_sub(m_.template triangular_view<Upper>(), x);
        return x;
    }
};

}


#endif   // _FDAPDE_MATRIX_H__