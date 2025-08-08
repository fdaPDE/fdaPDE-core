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

#ifndef __FDAPDE_SKEW_SYMMETRIC_MATRIX_H__
#define __FDAPDE_SKEW_SYMMETRIC_MATRIX_H__

#include "header_check.h"

namespace fdapde {

// forward declaration
template <typename Scalar_, int N_> class SkewSymmetricMatrixView;

// has_identity trait
namespace internals {

// SkewSymmetricMatrix => no identity
template <typename Scalar_, int N_, bool NestAsRefBit_>
struct has_identity<SkewSymmetricMatrix<Scalar_, N_, NestAsRefBit_>> : std::false_type {};

}

// is_view trait
namespace internals {

template <typename Scalar_, int N_>
struct is_view<SkewSymmetricMatrixView<Scalar_, N_>> : std::true_type {};

}

// symmetric matrix view
template <typename Scalar_, int N_>
class SkewSymmetricMatrixView : public SquareMatrixBase<N_, SkewSymmetricMatrixView<Scalar_, N_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = SkewSymmetricMatrixView<Scalar_, N_>;
    using Base = SquareMatrixBase<N_, MatrixViewType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ - 1) / 2;
    static constexpr bool NestAsRefBit = true;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::skew_symmetric);

    // constructors
    constexpr SkewSymmetricMatrixView() = delete;
    constexpr explicit SkewSymmetricMatrixView(Scalar* ptr_data) : ptr_data_(ptr_data) {}
    constexpr explicit SkewSymmetricMatrixView(std::array<Scalar, StorageSize>& data) : ptr_data_(data.data()) {}
    constexpr SkewSymmetricMatrixView(const MatrixViewType& other) : ptr_data_(other.ptr_data_) { }

    // copy operator
    constexpr MatrixViewType& operator=(const MatrixViewType& other) {
        if (this == &other) return *this;
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = other.data()[id];
        }
        return *this;
    }

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
            ptr_data_[id] = Scalar(0.5) * (rhs.derived()(i, j) - rhs.derived()(j, i));
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
                ptr_data_[id] = Scalar(0.5) * (rhs.derived()(i, j) - rhs.derived()(j, i));
            }
            return *this;
        }
    #endif

    // const access
    constexpr Scalar operator()(const int i, const int j) const {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols);
        if (j == i) return 0;
        return (j > i) ? ptr_data_[index(i, j)] : -ptr_data_[index(j, i)];
    }
    /*
    // non-const access
    static inline Scalar dummy_ref_ = Scalar(0);
    constexpr Scalar& operator()(const int i, const int j) {
        assert(i >= 0 && i < N && j >= 0 && j < N && j >= i);
        if (j > i) return ptr_data_[index(i, j)];
        if (i == j) return dummy_ref_;
        return dummy_ref_; // check this
    }
    */

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
    static constexpr int index(int i, int j) {
        assert(i < N && j > i && j < N);
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


// SkewSymmetricMatrix = SkewSymmetricMatrixView + data ownership
template <typename Scalar_, int N_, bool NestAsRefBit_ = true>
class SkewSymmetricMatrix : public SquareMatrixBase<N_, SkewSymmetricMatrix<Scalar_, N_, NestAsRefBit_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = SkewSymmetricMatrixView<Scalar_, N_>;
    using MatrixType = SkewSymmetricMatrix<Scalar_, N_, NestAsRefBit_>;
    using Base = SquareMatrixBase<N_, MatrixType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ - 1) / 2;
    static constexpr bool NestAsRefBit = NestAsRefBit_;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::skew_symmetric);

    // default constructor
    constexpr SkewSymmetricMatrix() : data_(), m_(data_.data()) { };

    // copy constructor
    constexpr SkewSymmetricMatrix(const MatrixType& other) : data_(), m_(data_.data()) { m_ = other; }

    // copy operator
    constexpr MatrixType& operator=(const MatrixType& other) { m_ = other; return *this; }

    // constructor from std::array
    constexpr explicit SkewSymmetricMatrix(const std::array<Scalar, StorageSize>& arr) : SkewSymmetricMatrix() { m_ = arr; }

    // constructor from C-style array
    constexpr explicit SkewSymmetricMatrix(const Scalar_ (&arr)[StorageSize]) : SkewSymmetricMatrix() { m_ = arr; }

    // constructor from std::vector
    constexpr explicit SkewSymmetricMatrix(const std::vector<Scalar>& vec) : SkewSymmetricMatrix() { m_ = vec; }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit SkewSymmetricMatrix(Callable callable) : SkewSymmetricMatrix() { m_ = callable; }

    // copy constructor from any MatrixBase-derived expression (templated)
    template<int OtherRows_, int OtherCols_, typename OtherDerived>
    constexpr explicit SkewSymmetricMatrix(const MatrixBase<OtherRows_,OtherCols_,OtherDerived>& xpr) : SkewSymmetricMatrix() { m_ = xpr; }

    // conversion constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template<typename Derived>
        explicit SkewSymmetricMatrix(const Eigen::MatrixBase<Derived>& other) : SkewSymmetricMatrix() { m_ = other; }
    #endif


    // static named constructors
    static constexpr SkewSymmetricMatrix Constant(Scalar c) {
        std::array<Scalar, StorageSize> data{};
        for (auto& val : data) val = c;
        return SkewSymmetricMatrix(data);
    }
    static constexpr SkewSymmetricMatrix Zero() { return Constant(Scalar(0)); }
    static constexpr SkewSymmetricMatrix Ones() { return Constant(Scalar(1)); }
    static constexpr SkewSymmetricMatrix NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // assignment from std::array
    constexpr SkewSymmetricMatrix& operator=(const std::array<Scalar, StorageSize>& rhs) { m_ = rhs; return *this; }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr SkewSymmetricMatrix& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) { m_ = rhs; return *this; }

    // assignment from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        SkewSymmetricMatrix& operator=(const Eigen::MatrixBase<Derived>& rhs) { m_ = rhs; return *this; }
    #endif

    // const access
    constexpr Scalar operator()(int i, int j) const { return m_(i, j); }
    constexpr Scalar operator[](int i) const requires(Rows == 1 || Cols == 1) { return m_[i]; }
    /*
    // non-const access
    constexpr Scalar& operator()(int i, int j) { return m_(i, j); }
    constexpr Scalar& operator[](int i) requires(Rows == 1 || Cols == 1)  { return m_[i]; }
    */

    // convert to full matrix
    constexpr Matrix<Scalar, N, N> as_matrix() const {
        Matrix<Scalar, N, N> M;
        for (int i = 0; i < N; ++i) {
            M(i, i) = 0;
            for (int j = i + 1; j < N; ++j) {
                M(i, j) = (*this)(i, j);
                M(j, i) = -(*this)(i, j);
            }
        }
        return M;
    }

    // convert to EigenMap
    #ifdef __FDAPDE_HAS_EIGEN__
        // TODO: differently from Matrix here I can not return a map because the Map saves the pointer to data_.data() but symmetric matrices stores data in a non compatible way
        auto as_eigen() {
            Matrix<Scalar, Rows, Cols, RowMajor, NestAsRefBit> M(as_matrix());
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
    SkewSymmetricMatrixView<Scalar, N> m_;

};

}

#endif // __FDAPDE_SKEW_SYMMETRIC_MATRIX_H__