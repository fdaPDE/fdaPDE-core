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

#ifndef __FDAPDE_SYMMETRIC_MATRIX_H__
#define __FDAPDE_SYMMETRIC_MATRIX_H__

#include "header_check.h"
#include "matrix_base.h"
#include "square_matrix_base.h"
#include "symmetric_matrix.h"

namespace fdapde {

// has_identity trait
namespace internals {

// SymmetricMatrix => has identity
template <typename Scalar_, int N_, bool NestAsRefBit_>
struct has_identity<SymmetricMatrix<Scalar_, N_, NestAsRefBit_>> : std::true_type {};

}

// symmetric matrix view
template <typename Scalar_, int N_, int StorageOrder_>
struct SymmetricMatrixView : public MatrixView<Scalar_, N_, N_, StorageOrder_> {

    using ViewType = SymmetricMatrixView<Scalar_, N_, StorageOrder_>;
    using Base = MatrixView<Scalar_, N_, N_, StorageOrder_>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * N_;
    static constexpr bool NestAsRefBit = true;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::symmetric);

    // constructors
    constexpr SymmetricMatrixView() = default;
    constexpr explicit SymmetricMatrixView(Scalar_* data) : Base(data) {}

    // const access
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i < Rows && j < Cols);
        return 0.5*(Base::operator()(i, j) + Base::operator()(j, i) ) ;
    }
};

// symmetric matrix
template <typename Scalar_, int N_, bool NestAsRefBit_ = true>
class SymmetricMatrix : public SquareMatrixBase<N_, SymmetricMatrix<Scalar_, N_, NestAsRefBit_>> {

public:
    using Base = SquareMatrixBase<N_, SymmetricMatrix<Scalar_, N_, NestAsRefBit_>>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ + 1) / 2;
    static constexpr bool NestAsRefBit = NestAsRefBit_;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::symmetric);

    // default constructor
    constexpr SymmetricMatrix() : data_() {};

    // copy constructor (defaulted)
    constexpr SymmetricMatrix(const SymmetricMatrix& other) = default;

    // move constructor (defaulted)
    constexpr SymmetricMatrix(SymmetricMatrix&& other) = default;

    // constructor from std::array
    constexpr explicit SymmetricMatrix(const std::array<Scalar, StorageSize>& data) : data_(data) { }

    // constructor from C-style array
    constexpr explicit SymmetricMatrix(const Scalar_ (&data)[StorageSize]) : data_() {
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = data[id];
        }
    }

    // constructor from std::vector
    constexpr explicit SymmetricMatrix(const std::vector<Scalar>& data) : data_() {
        fdapde_constexpr_assert(data.size() == StorageSize);
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = data[id];
        }
    }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit SymmetricMatrix(Callable callable)
        requires(std::is_invocable_v<Callable>)
        : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {
            callable})::result_type FDAPDE_COMMA std::array<Scalar FDAPDE_COMMA StorageSize>>,
          CALLABLE_DOES_NOT_RETURN_SOMETHING_CONVERTIBLE_TO_AN_ARRAY_OF_SCALAR);
        data_ = callable();
    }

    // converting constructor from another Matrix expression
    template <typename Derived>
    constexpr explicit SymmetricMatrix(const MatrixBase<Rows, Cols, Derived>& xpr) : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_TYPES_CONVERSION_BETWEEN_MATRICES);
        fdapde_static_assert(
          Derived::Rows == Rows && Derived::Cols == Cols,
          YOU_ARE_TRYING_TO_CONSTRUCT_A_MATRIX_WITH_ANOTHER_MATRIX_OF_DIFFERENT_SIZE);
        for (int id = 0; id < StorageSize; ++id) {
            auto [i, j] = inv_index(id);
            data_[id] = Scalar(0.5) * (xpr.derived()(i, j) + xpr.derived()(j, i));
        }
    }

    // converting constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        constexpr explicit SymmetricMatrix(const Eigen::MatrixBase<Derived>& other) {
            constexpr int Rows__ = Derived::RowsAtCompileTime;
            constexpr int Cols__ = Derived::ColsAtCompileTime;
            fdapde_static_assert(
              Rows__ != Dynamic && Cols__ != Dynamic && Rows__ == Rows && Cols__ == Cols &&
                std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
              INVALID_CONVERSION_FROM_EIGEN_MATRIX_TO_FDAPDE_MATRIX);
            for (int id = 0; id < StorageSize; ++id) {
                auto [i, j] = inv_index(id);
                data_[id] = Scalar(0.5) * (other.derived()(i, j) + other.derived()(j, i));
            }
        }
    #endif

    // static named constructors
    static constexpr SymmetricMatrix<Scalar, N> Constant(Scalar c) {
        std::array<Scalar, StorageSize> data{};
        for (auto& val : data) val = c;
        return SymmetricMatrix<Scalar, N>(data);
    }
    static constexpr SymmetricMatrix<Scalar, N> Zero() { return Constant(Scalar(0)); }
    static constexpr SymmetricMatrix<Scalar, N> Ones() { return Constant(Scalar(1)); }
    static constexpr SymmetricMatrix<Scalar, N> NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // copy assignment operator
    constexpr SymmetricMatrix& operator=(const SymmetricMatrix& other) = default;

    // assignment from std::array
    constexpr SymmetricMatrix& operator=(const std::array<Scalar, StorageSize>& rhs) {
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = rhs[id];
        }
        return *this;
    }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr SymmetricMatrix& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_DIMENSIONS_OR_YOU_ARE_TRYING_TO_ASSIGN_A_RHS_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        *this = SymmetricMatrix(rhs);
        return *this;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // assignment from Eigen matrix
        template <typename Derived>
        SymmetricMatrix& operator=(const Eigen::MatrixBase<Derived>& rhs) {
            fdapde_static_assert(
              Derived::RowsAtCompileTime != Dynamic && Derived::ColsAtCompileTime != Dynamic &&
                std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
              CANNOT_ASSIGN_FROM_EIGEN_HEAP_ALLOCATED_MATRIX_OR_INVALID_SCALAR_TYPE);
            *this = SymmetricMatrix(rhs);
            return *this;
        }
    #endif

    // const access
    constexpr Scalar operator()(int i, int j) const{
        assert(i >= 0 && i < N && j >= 0 && j < N);
        return (j >= i) ? data_[index(i, j)] : data_[index(j, i)];
    }
    // non-const access
    constexpr Scalar& operator()(int i, int j){
        assert(i >= 0 && i < N && j >= 0 && j < N);
        return (j >= i) ? data_[index(i, j)] : data_[index(j, i)];
    }

    // convert to full matrix
    constexpr fdapde::Matrix<Scalar_, N, N> full() const {
        fdapde::Matrix<Scalar_, N, N> M;
        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                M(i, j) = (*this)(i, j);
                M(j, i) = (*this)(i, j);
            }
        }
        return M;
    }

    // convert to Eigen
    // TODO: differently from Matrix here I can not return a map because the Map saves the pointer to data_.data() but symmetric matrices stores data in a non compatible way
    #ifdef __FDAPDE_HAS_EIGEN__
        Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor> as_eigen() const {
            Matrix<Scalar, Rows, Cols, RowMajor, NestAsRefBit> M(this->full());
            return Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor>(M.data());
        }
    #endif

    // data
    constexpr const Scalar* data() const { return data_.data(); }
    Scalar* data() { return data_.data(); }
    constexpr const std::array<Scalar,StorageSize>& storage() const { return data_; }

    // setters
    constexpr void setConstant(Scalar c) {
        for (int id = 0; id < StorageSize; ++id) data_[id] = c;
    }
    constexpr void setZero() { setConstant(Scalar(0)); }
    constexpr void setOnes() { setConstant(Scalar(1)); }

private:
    std::array<Scalar, StorageSize> data_;

    static constexpr int index(int i, int j){
        assert(j >= i && j < N);
        // assumes j >= i
        return i * (2 * N - i + 1) / 2 + (j - i);
    }
    static constexpr std::pair<int, int> inv_index(int idx){
        assert(idx >= 0 && idx < StorageSize);
        int i = 0;
        int offset = 0;
        // loop to find the row i such that idx is within its block
        while (i < N) {
            int row_len = N - i;
            if (idx < offset + row_len)
                break;
            offset += row_len;
            ++i;
        }
        int j = i + (idx - offset);
        return {i, j};
    }
};

}

#endif // __FDAPDE_SYMMETRIC_MATRIX_H__