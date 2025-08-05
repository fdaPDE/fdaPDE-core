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

#ifndef __FDAPDE_SKEW_SYMMETRIC_MATRIX_BASE_H__
#define __FDAPDE_SKEW_SYMMETRIC_MATRIX_BASE_H__

#include "header_check.h"
#include "matrix_base.h"

namespace fdapde {

// has_identity trait
namespace internals {

// SkewSymmetricMatrix => no identity
template <typename Scalar_, int N_, bool NestAsRefBit_>
struct has_identity<SkewSymmetricMatrix<Scalar_, N_, NestAsRefBit_>> : std::false_type {};

}

// symmetric matrix view
template <typename Scalar_, int N_, int StorageOrder_>
struct SkewSymmetricMatrixView : public MatrixView<Scalar_, N_, N_, StorageOrder_> {

    using ViewType = SkewSymmetricMatrixView<Scalar_, N_, StorageOrder_>;
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
    constexpr SkewSymmetricMatrixView() = default;
    constexpr explicit SkewSymmetricMatrixView(Scalar_* data) : Base(data) {}

    // const access
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i < Rows && j < Cols);
        return 0.5*(Base::operator()(i, j) + Base::operator()(j, i) ) ;
    }
};

// symmetric matrix
template <typename Scalar_, int N_, bool NestAsRefBit_ = true>
class SkewSymmetricMatrix : public SquareMatrixBase<N_, SkewSymmetricMatrix<Scalar_, N_, NestAsRefBit_>> {
public:
    using Base = SquareMatrixBase<N_, SkewSymmetricMatrix<Scalar_, N_, NestAsRefBit_>>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ - 1) / 2;
    static constexpr bool NestAsRefBit = NestAsRefBit_;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::skew_symmetric);

    // default constructor
    constexpr SkewSymmetricMatrix() : data_() {}

    // copy constructor (defaulted)
    constexpr SkewSymmetricMatrix(const SkewSymmetricMatrix& other) = default;

    // move constructor (defaulted)
    constexpr SkewSymmetricMatrix(SkewSymmetricMatrix& other) = default;

    // constructor from std::array
    constexpr explicit SkewSymmetricMatrix(const std::array<Scalar, StorageSize>& data) : data_(data) {}

    // constructor from C-style array
    constexpr explicit SkewSymmetricMatrix(const Scalar_ (&data)[StorageSize]) : data_() {
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = data[id];
        }
    }

    // constructor from std::vector
    constexpr explicit SkewSymmetricMatrix(const std::vector<Scalar>& data) : data_() {
        fdapde_constexpr_assert(data.size() == StorageSize);
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = data[id];
        }
    }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit SkewSymmetricMatrix(Callable callable)
        requires(std::is_invocable_v<Callable>)
        : data_() {
        fdapde_static_assert(
            std::is_convertible_v<typename decltype(std::function { callable })::result_type
                                  FDAPDE_COMMA std::array<Scalar FDAPDE_COMMA StorageSize>>,
            CALLABLE_DOES_NOT_RETURN_SOMETHING_CONVERTIBLE_TO_AN_ARRAY_OF_SCALAR);
        data_ = callable();
    }

    // converting constructor from another Matrix expression
    template <typename Derived>
    constexpr explicit SkewSymmetricMatrix(const MatrixBase<Rows, Cols, Derived>& xpr) : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_TYPES_CONVERSION_BETWEEN_MATRICES);
        fdapde_static_assert(
          Derived::Rows == Rows && Derived::Cols == Cols,
          YOU_ARE_TRYING_TO_CONSTRUCT_A_MATRIX_WITH_ANOTHER_MATRIX_OF_DIFFERENT_SIZE);
        for (int id = 0; id < StorageSize; ++id) {
            auto [i, j] = inv_index(id);
            data_[id] = Scalar(0.5) * (xpr.derived()(i, j) - xpr.derived()(j, i));
        }
    }

    // converting constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        constexpr explicit SkewSymmetricMatrix(const Eigen::MatrixBase<Derived>& other) {
            constexpr int Rows__ = Derived::RowsAtCompileTime;
            constexpr int Cols__ = Derived::ColsAtCompileTime;
            fdapde_static_assert(
              Rows__ != Dynamic && Cols__ != Dynamic && Rows__ == Rows && Cols__ == Cols &&
                std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
              INVALID_CONVERSION_FROM_EIGEN_MATRIX_TO_FDAPDE_MATRIX);
            for (int id = 0; id < StorageSize; ++id) {
                auto [i, j] = inv_index(id);
                data_[id] = Scalar(0.5) * (other.derived()(i, j) - other.derived()(j, i));
            }
        }
    #endif

    // static named constructors
    static constexpr SkewSymmetricMatrix<Scalar, N> Constant(Scalar c) {
        std::array<Scalar, StorageSize> data{};
        for (auto& val : data) val = c;
        SkewSymmetricMatrix<Scalar, N> m(data);
        return m;
    }
    static constexpr SkewSymmetricMatrix<Scalar, N> Zero() { return Constant(Scalar(0)); }
    static constexpr SkewSymmetricMatrix<Scalar, N> Ones() { return Constant(Scalar(1)); }
    static constexpr SkewSymmetricMatrix<Scalar, N> NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // copy assignment operator
    constexpr SkewSymmetricMatrix& operator=(const SkewSymmetricMatrix& other) = default;

    // assignment from std::array
    constexpr SkewSymmetricMatrix& operator=(const std::array<Scalar, StorageSize>& rhs) {
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = rhs[id];
        }
        return *this;
    }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr SkewSymmetricMatrix& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_DIMENSIONS_OR_YOU_ARE_TRYING_TO_ASSIGN_A_RHS_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        *this = SkewSymmetricMatrix(rhs);
        return *this;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // assignment from Eigen matrix
        template <typename Derived>
        SkewSymmetricMatrix& operator=(const Eigen::MatrixBase<Derived>& rhs) {
            fdapde_static_assert(
              Derived::RowsAtCompileTime != Dynamic && Derived::ColsAtCompileTime != Dynamic &&
                std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
              CANNOT_ASSIGN_FROM_EIGEN_HEAP_ALLOCATED_MATRIX_OR_INVALID_SCALAR_TYPE);
            *this = SkewSymmetricMatrix(rhs);
            return *this;
        }
    #endif

    // const access
    constexpr Scalar operator()(int i, int j) const {
        assert(i >= 0 && i < N && j >= 0 && j < N);
        if (i == j) return Scalar(0);
        return (j > i) ? data_[index(i, j)] : -data_[index(j, i)];
    }
    // non-const access
    static inline Scalar dummy_ref_ = Scalar(0);
    constexpr Scalar& operator()(int i, int j) {
        assert(i >= 0 && i < N && j >= 0 && j < N);
        if (j > i) return data_[index(i, j)];
        if (i == j) return dummy_ref_;
        return dummy_ref_; // check this
    }

    // convert to full matrix
    constexpr fdapde::Matrix<Scalar_, N, N> full() const {
        fdapde::Matrix<Scalar_, N, N> M;
        for (int i = 0; i < N; ++i) {
            M(i, i) = Scalar(0);
            for (int j = i + 1; j < N; ++j) {
                Scalar v = (*this)(i, j);
                M(i, j) = v;
                M(j, i) = -v;
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

private:
    std::array<Scalar, StorageSize> data_;

    static constexpr int index(int i, int j) {
        assert(j > i && j < N);
        return i * (2 * N - i - 1) / 2 + (j - i - 1);
    }
    static constexpr std::pair<int, int> inv_index(int idx) {
        assert(idx >= 0 && idx < StorageSize);
        int i = 0;
        int offset = 0;
        // loop to find the row i such that idx is within its block
        while (i < N - 1) {
            int row_len = N - i - 1;
            if (idx < offset + row_len)
                break;
            offset += row_len;
            ++i;
        }
        int j = i + 1 + (idx - offset);
        return {i, j};
    }
};

} // namespace fdapde

#endif // __FDAPDE_SKEW_SYMMETRIC_MATRIX_BASE_H__