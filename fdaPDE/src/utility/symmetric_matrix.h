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

// forward declaration
template <typename Scalar_, int N_> class SymmetricMatrixView;

// has_identity trait
namespace internals {

// SymmetricMatrix => has identity
template <typename Scalar_, int N_, bool NestAsRefBit_>
struct has_identity<SymmetricMatrix<Scalar_, N_, NestAsRefBit_>> : std::true_type {};

}

// is_view trait
namespace internals {

template <typename Scalar_, int N_>
struct is_view<SymmetricMatrixView<Scalar_, N_>> : std::true_type {};

}

// symmetric matrix view
template <typename Scalar_, int N_>
class SymmetricMatrixView : public SquareMatrixBase<N_, SymmetricMatrixView<Scalar_, N_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = SymmetricMatrixView<Scalar_, N_>;
    using Base = SquareMatrixBase<N_, MatrixViewType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ + 1) / 2;
    static constexpr bool NestAsRefBit = true;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::symmetric);

    // constructors
    constexpr SymmetricMatrixView() = delete;
    constexpr explicit SymmetricMatrixView(Scalar* ptr_data) : ptr_data_(ptr_data) {}
    constexpr explicit SymmetricMatrixView(std::array<Scalar, StorageSize>& data) : ptr_data_(data.data()) {}

    // assignment from std::array
    constexpr MatrixViewType& operator=(const std::array<Scalar, StorageSize>& rhs) {
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = rhs[id];
        }
        return *this;
    }

    // assignment from C-array
    constexpr MatrixViewType& operator=(const Scalar (&rhs)[StorageSize]) {
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = rhs[id];
        }
        return *this;
    }

    // assignment from std::vector
    constexpr MatrixViewType& operator=(const std::vector<Scalar>& rhs) {
        fdapde_constexpr_assert(rhs.size() == StorageSize);
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = rhs[id];
        }
        return *this;
    }

    // assignment from any callable returning std::array<Scalar, StorageSize>
    template <typename Callable>
    constexpr MatrixViewType& operator=(Callable callable)
    requires(std::is_invocable_v<Callable>) {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {
            callable})::result_type FDAPDE_COMMA std::array<Scalar FDAPDE_COMMA StorageSize>>,
          CALLABLE_DOES_NOT_RETURN_SOMETHING_CONVERTIBLE_TO_AN_ARRAY_OF_SCALAR);
        *this = callable();
        return *this;
    }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr MatrixViewType& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_DIMENSIONS_OR_YOU_ARE_TRYING_TO_ASSIGN_A_RHS_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int id = 0; id < StorageSize; ++id) {
            auto[i, j] = inv_index(id);
            ptr_data_[id] = Scalar(0.5) * (rhs.derived()(i, j) + rhs.derived()(j, i));
        }
        return *this;
    }

    // assignment from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename OtherDerived>
        MatrixViewType& operator=(const Eigen::MatrixBase<OtherDerived>& rhs) {
            fdapde_static_assert(
              OtherDerived::RowsAtCompileTime != Dynamic && OtherDerived::ColsAtCompileTime != Dynamic &&
                std::is_convertible_v<typename OtherDerived::Scalar FDAPDE_COMMA Scalar>,
              CANNOT_ASSIGN_FROM_EIGEN_HEAP_ALLOCATED_MATRIX_OR_INVALID_SCALAR_TYPE);
            for (int id = 0; id < StorageSize; ++id) {
                auto[i, j] = inv_index(id);
                ptr_data_[id] = Scalar(0.5) * (rhs.derived()(i, j) + rhs.derived()(j, i));
            }
            return *this;
        }
    #endif

    // const access
    constexpr Scalar operator()(const int i, const int j) const {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols);
        return (j >= i) ? ptr_data_[index(i, j)] : ptr_data_[index(j, i)];
    }
    constexpr Scalar operator[](const int i) const
    requires(Cols == 1 || Rows == 1) {
        fdapde_assert(i >= 0 && i < StorageSize);
        return ptr_data_[i];
    }
    // non-const access
    constexpr Scalar& operator()(const int i, const int j) {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols);
        return (j >= i) ? ptr_data_[index(i, j)] : ptr_data_[index(j, i)];
    }
    constexpr Scalar& operator[](const int i)
    requires(Cols == 1 || Rows == 1) {
        fdapde_assert(i >= 0 && i < StorageSize);
        return ptr_data_[i];
    }

    // setters
    constexpr void setConstant(Scalar c) {
        for (int id = 0; id < StorageSize; ++id) ptr_data_[id] = c;
    }
    constexpr void setZero() { setConstant(Scalar(0)); }
    constexpr void setOnes() { setConstant(Scalar(1)); }
    constexpr void setNaN() { setConstant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // dimensions
    static constexpr int rows() { return Rows; }
    static constexpr int cols() { return Cols; }

    // data
    constexpr const Scalar_* data() const { return ptr_data_; }
    constexpr Scalar_* data() { return ptr_data_; }

protected:
    Scalar_* ptr_data_ = nullptr;

    // indexes
    static constexpr int index(int i, int j){
        assert(j >= i && j < N);
        // assumes j >= i
        return i * (2 * N - i + 1) / 2 + (j - i);
    }
    static constexpr std::pair<int, int> inv_index(int idx) {
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


// SymmetricMatrix = SymmetricMatrixView + data ownership
template <typename Scalar_, int N_, bool NestAsRefBit_ = true>
class SymmetricMatrix : public SquareMatrixBase<N_, SymmetricMatrix<Scalar_, N_, NestAsRefBit_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = SymmetricMatrixView<Scalar_, N_>;
    using MatrixType = SymmetricMatrix<Scalar_, N_, NestAsRefBit_>;
    using Base = SquareMatrixBase<N_, MatrixType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ + 1) / 2;
    static constexpr bool NestAsRefBit = NestAsRefBit_;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::symmetric);

    // default constructor
    constexpr SymmetricMatrix() : data_(), m_(data_.data()) { };

    // constructor from std::array
    constexpr explicit SymmetricMatrix(const std::array<Scalar, StorageSize>& arr) : SymmetricMatrix() { m_ = arr; }

    // constructor from C-style array
    constexpr explicit SymmetricMatrix(const Scalar_ (&arr)[StorageSize]) : SymmetricMatrix() { m_ = arr; }

    // constructor from std::vector
    constexpr explicit SymmetricMatrix(const std::vector<Scalar>& vec) : SymmetricMatrix() { m_ = vec; }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit SymmetricMatrix(Callable callable) : SymmetricMatrix() { m_ = callable; }

    // copy constructor from any MatrixBase-derived expression (templated)
    template<int OtherRows_, int OtherCols_, typename OtherDerived>
    constexpr explicit SymmetricMatrix(const MatrixBase<OtherRows_,OtherCols_,OtherDerived>& xpr) : SymmetricMatrix() { m_ = xpr; }

    // conversion constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template<typename Derived>
        explicit SymmetricMatrix(const Eigen::MatrixBase<Derived>& other) : SymmetricMatrix() { m_ = other; }
    #endif


    // static named constructors
    static constexpr SymmetricMatrix Constant(Scalar c) {
        std::array<Scalar, StorageSize> data{};
        for (auto& val : data) val = c;
        return SymmetricMatrix(data);
    }
    static constexpr SymmetricMatrix Zero() { return Constant(Scalar(0)); }
    static constexpr SymmetricMatrix Ones() { return Constant(Scalar(1)); }
    static constexpr SymmetricMatrix NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // assignment from std::array
    constexpr SymmetricMatrix& operator=(const std::array<Scalar, StorageSize>& rhs) { m_ = rhs; return *this; }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr SymmetricMatrix& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) { m_ = rhs; return *this; }

    // assignment from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        SymmetricMatrix& operator=(const Eigen::MatrixBase<Derived>& rhs) { m_ = rhs; return *this; }
    #endif

    // const access
    constexpr Scalar operator()(int i, int j) const { return m_(i, j); }
    constexpr Scalar operator[](int i) const requires(Rows == 1 || Cols == 1) { return m_[i]; }
    // non-const access
    constexpr Scalar& operator()(int i, int j) { return m_(i, j); }
    constexpr Scalar& operator[](int i) requires(Rows == 1 || Cols == 1)  { return m_[i]; }

    // convert to full matrix
    constexpr Matrix<Scalar_, N, N> full() const {
        Matrix<Scalar_, N, N> M;
        for (int i = 0; i < N; ++i)
            for (int j = i; j < N; ++j)
                M(j, i) = M(i, j) = (*this)(i, j);
        return M;
    }

    // convert to EigenMap
    #ifdef __FDAPDE_HAS_EIGEN__
        // TODO: differently from Matrix here I can not return a map because the Map saves the pointer to data_.data() but symmetric matrices stores data in a non compatible way
        auto as_eigen() {
            Matrix<Scalar, Rows, Cols, RowMajor, NestAsRefBit> M(full());
            return Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor>(M.data());
        }
    #endif

    // setters
    constexpr void setConstant(Scalar c) { m_.setConstant(c); }
    constexpr void setZero() { setConstant(Scalar(0)); }
    constexpr void setOnes() { setConstant(Scalar(1)); }
    constexpr void setNaN() { setConstant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // dimensions
    static constexpr int rows() { return Rows; }
    static constexpr int cols() { return Cols; }

    // data
    constexpr const Scalar* data() const { return m_.data(); }
    Scalar* data() { return m_.data(); }
    constexpr const std::array<Scalar,StorageSize>& storage() const { return data_; }

   private:
    std::array<Scalar, StorageSize> data_;
    SymmetricMatrixView<Scalar, N> m_;

};

}

#endif // __FDAPDE_SYMMETRIC_MATRIX_H__