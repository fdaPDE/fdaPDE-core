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

namespace fdapde {

template <typename Scalar_, int Rows_, int Cols_, int NestAsRefBit_ = 1>
class Matrix : public std::conditional_t<Rows_ == Cols_, SquareMatrixBase<Rows_, Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>>, MatrixBase<Rows_, Cols_, Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>>> {
    fdapde_static_assert(Rows_ > 0 && Cols_ > 0, EMPTY_MATRIX_IS_ILL_FORMED);
   public:
    using Base = std::conditional_t<Rows_ == Cols_, SquareMatrixBase<Rows_, Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>>, MatrixBase<Rows_, Cols_, Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>>>;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageSize = Rows_ * Cols_;
    static constexpr int NestAsRefBit = NestAsRefBit_;   // whether to store this node by ref or by copy in an expression
    static constexpr int ReadOnly = 0;
    static constexpr int XprBits = (Rows_ == Cols_) ? int(matrix_flags::square) : int(matrix_flags::none);

    // default constructor
    constexpr Matrix() : data_() {};

    // constructor from std::array
    constexpr explicit Matrix(const std::array<Scalar, StorageSize>& data) : data_(data) { }

    // constructor from C-style array
    constexpr explicit Matrix(const Scalar_ (&data)[StorageSize]) : data_() {
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = data[id];
        }
    }

    // constructor from std::vector
    constexpr explicit Matrix(const std::vector<Scalar>& data) : data_() {
        fdapde_constexpr_assert(data.size() == StorageSize);
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = data[id];
        }
    }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit Matrix(Callable callable)
        requires(std::is_invocable_v<Callable>)
        : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {
            callable})::result_type FDAPDE_COMMA std::array<Scalar FDAPDE_COMMA StorageSize>>,
          CALLABLE_DOES_NOT_RETURN_SOMETHING_CONVERTIBLE_TO_AN_ARRAY_OF_SCALAR);
        data_ = callable();
    }

    // copy constructor from any MatrixBase-derived expression (templated)
    template <typename Derived>
    constexpr Matrix(const MatrixBase<Rows, Cols, Derived>& xpr) : data_() { // this must not be explicit
        fdapde_static_assert(
          std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_TYPES_CONVERSION_BETWEEN_MATRICES);
        fdapde_static_assert(
          Derived::Rows == Rows && Derived::Cols == Cols,
          YOU_ARE_TRYING_TO_CONSTRUCT_A_MATRIX_WITH_ANOTHER_MATRIX_OF_DIFFERENT_SIZE);
        for (int id = 0; id < StorageSize; ++id) {
            auto[i, j] = inv_index(id);
            data_[id] = xpr.derived().operator()(i, j);
        }
    }

    // conversion constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived> Matrix(const Eigen::MatrixBase<Derived>& other) {
            constexpr int Rows__ = Derived::RowsAtCompileTime;
            constexpr int Cols__ = Derived::ColsAtCompileTime;
            fdapde_static_assert(
              Rows__ != Dynamic && Cols__ != Dynamic && Rows__ == Rows && Cols__ == Cols &&
                std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
              INVALID_CONVERSION_FROM_EIGEN_MATRIX_TO_FDAPDE_MATRIX);
            for (int id = 0; id < StorageSize; ++id) {
                auto[i, j] = inv_index(id);
                data_[id] = other(i, j);
            }
        }
    #endif

    // scalar constructor for 1D vector/matrix (StorageSize == 1)
    constexpr explicit Matrix(Scalar x) : data_() {   // 1D point constructor
        fdapde_static_assert(StorageSize == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_ONE_ELEMENT);
        data_[0] = x;
    }

    // constructor for 2D vector (StorageSize == 2)
    constexpr explicit Matrix(Scalar x, Scalar y) : data_() {   // 2D point constructor
        fdapde_static_assert(StorageSize == 2, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_TWO_ELEMENTS);
        data_ = {x, y};
    }

    // constructor for 3D vector (StorageSize == 3)
    constexpr explicit Matrix(Scalar x, Scalar y, Scalar z) : data_() {   // 3D point constructor
        fdapde_static_assert(StorageSize == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_THREE_ELEMENTS);
        data_ = {x, y, z};
    }

    // assignment from std::array
    constexpr Matrix<Scalar, Rows, Cols, NestAsRefBit>& operator=(const std::array<Scalar, StorageSize>& rhs) {
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = rhs[id];
        }
        return *this;
    }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr Matrix<Scalar, Rows, Cols, NestAsRefBit>&
    operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_DIMENSIONS_OR_YOU_ARE_TRYING_TO_ASSIGN_A_RHS_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int id = 0; id < StorageSize; ++id) {
            auto[i, j] = inv_index(id);
            data_[id] = rhs.derived()(i, j);
        }
        return *this;
    }

    // assignment from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        Matrix<Scalar, Rows, Cols, NestAsRefBit>& operator=(const Eigen::MatrixBase<Derived>& rhs) {
            fdapde_static_assert(
              Derived::RowsAtCompileTime != Dynamic && Derived::ColsAtCompileTime != Dynamic &&
                std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
              CANNOT_ASSIGN_FROM_EIGEN_HEAP_ALLOCATED_MATRIX_OR_INVALID_SCALAR_TYPE);
            for (int id = 0; id < StorageSize; ++id) {
                auto[i, j] = inv_index(id);
                data_[id] = rhs.derived()(i, j);
            }
            return *this;
        }
    #endif

    // static named constructors
    static constexpr Matrix<Scalar, Rows, Cols> Constant(Scalar c) {
        std::array<Scalar, StorageSize> data{};
        for (auto& val : data) val = c;
        return Matrix<Scalar, Rows, Cols>(data);
    }
    static constexpr Matrix<Scalar, Rows, Cols> Zero() { return Constant(Scalar(0)); }
    static constexpr Matrix<Scalar, Rows, Cols> Ones() { return Constant(Scalar(1)); }
    static constexpr Matrix<Scalar, Rows, Cols> NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // const access
    constexpr Scalar operator()(int i, int j) const { return data_[index(i, j)]; }
    constexpr Scalar operator[](int i) const
        requires(Cols == 1 || Rows == 1) {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_[i];
    }
    // non-const access
    constexpr Scalar& operator()(int i, int j) { return data_[index(i, j)]; }
    constexpr Scalar& operator[](int i)
        requires(Cols == 1 || Rows == 1){
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_[i];
    }

    // convert to EigenMap
    #ifdef __FDAPDE_HAS_EIGEN__
        Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor>> as_eigen_map() {
            return Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor>>(data_.data());
        }
    #endif

    // data
    constexpr const Scalar* data() const { return data_.data(); }
    Scalar* data() { return data_.data(); }

    // setters
    constexpr void setConstant(Scalar c) {
        for (auto& val : data_) val = c;
    }
    constexpr void setZero() { setConstant(Scalar(0)); }
    constexpr void setOnes() { setConstant(Scalar(1)); }

   private:
    std::array<Scalar, StorageSize> data_;

    static constexpr int index(int i, int j) {
        return i * Cols + j;
    }
    static constexpr std::pair<int, int> inv_index(int id) {
        int i = id / Cols;
        int j = id % Cols;
        return {i, j};
    }
};


}


#endif   // _FDAPDE_MATRIX_H__