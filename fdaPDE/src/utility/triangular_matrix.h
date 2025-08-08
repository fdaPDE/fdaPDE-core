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

#ifndef __FDAPDE_TRIANGULAR_MATRIX_H__
#define __FDAPDE_TRIANGULAR_MATRIX_H__

#include "header_check.h"
#include "matrix_base.h"
#include "square_matrix_base.h"
#include "triangular_matrix.h"

namespace fdapde {

// forward declaration
template <typename Scalar_, int N, int TriangularType> class TriangularMatrixView;
template <typename Scalar_, int N, int TriangularType, bool NestAsRefBit> class TriangularMatrix;

// alias for upper triangular matrices
template <typename Scalar_, int N, int TriangularType> using UpperTriangularMatrixView = TriangularMatrixView<Scalar_, N, Upper>;
template <typename Scalar_, int N, bool NestAsRefBit = true> using UpperTriangularMatrix = TriangularMatrix<Scalar_, N, Upper, NestAsRefBit>;

// alias for lower triangular matrices
template <typename Scalar_, int N, int TriangularType> using LowerTriangularMatrixView = TriangularMatrixView<Scalar_, N, Lower>;
template <typename Scalar_, int N, bool NestAsRefBit = true> using LowerTriangularMatrix = TriangularMatrix<Scalar_, N, Lower, NestAsRefBit>;

// has_identity trait
namespace internals {

// TriangularMatrix => has identity
template <typename Scalar_, int N, int TriangularType, bool NestAsRefBit>
struct has_identity<TriangularMatrix<Scalar_, N, TriangularType, NestAsRefBit>> : std::true_type {};

}

// is_view trait
namespace internals {

template <typename Scalar, int N, int TriangularType>
struct is_view<TriangularMatrixView<Scalar, N, TriangularType>> : std::true_type {};

}

// is_triangular trait
namespace internals {

template <typename Scalar, int N, int TriangularType>
struct is_triangular<TriangularMatrixView<Scalar, N, TriangularType>> : std::true_type {};
template <typename Scalar, int N, int TriangularType, bool NestAsRefBit>
struct is_triangular<TriangularMatrix<Scalar, N, TriangularType, NestAsRefBit>> : std::true_type {};

}

// triangular matrix view
template <typename Scalar_, int N_, int TriangularType_>
class TriangularMatrixView : public SquareMatrixBase<N_, TriangularMatrixView<Scalar_, N_, TriangularType_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = TriangularMatrixView<Scalar_, N_, TriangularType_>;
    using Base = SquareMatrixBase<N_, MatrixViewType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ + 1) / 2;
    static constexpr int TriangularType = TriangularType_;
    static constexpr bool NestAsRefBit = true;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | ((TriangularType == Upper) ? int(matrix_flags::upper_triangular) : int(matrix_flags::lower_triangular));

    // constructors
    constexpr TriangularMatrixView() = delete;
    constexpr explicit TriangularMatrixView(Scalar* ptr_data) : ptr_data_(ptr_data) {}
    constexpr explicit TriangularMatrixView(std::array<Scalar, StorageSize>& data) : ptr_data_(data.data()) {}
    constexpr TriangularMatrixView(const MatrixViewType& other) : ptr_data_(other.ptr_data_) { }

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

    // assignment from MatrixBase expression (take the selected triangular part)
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr MatrixViewType& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_DIMENSIONS_OR_YOU_ARE_TRYING_TO_ASSIGN_A_RHS_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int id = 0; id < StorageSize; ++id) {
            auto[i, j] = inv_index(id);
            ptr_data_[id] = rhs.derived()(i, j);
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
                ptr_data_[id] = rhs.derived()(i, j);
            }
            return *this;
        }
    #endif

    // const access
    constexpr Scalar operator()(const int i, const int j) const {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols);
        return in_triangle(i, j) ? ptr_data_[index(i, j)] : Scalar(0);
    }
    // non-const access
    static inline Scalar dummy_ref_ = Scalar(0);
    constexpr Scalar& operator()(const int i, const int j) {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols);
        if (in_triangle(i, j)) return ptr_data_[index(i, j)];
        return dummy_ref_;
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

    // helpers
    static constexpr bool in_triangle(int i, int j) {
        if constexpr (TriangularType_ == Upper) return j >= i;
        else                                     return i >= j;
    }

protected:
    Scalar_* ptr_data_ = nullptr;

    // indexes
    static constexpr int index(int i, int j){
        if constexpr (TriangularType_ == Upper) {
            assert(j >= i && j < N);
            // upper triangular (row-major, store from diagonal to end of row)
            return i * (2 * N - i + 1) / 2 + (j - i);
        } else {
            assert(i >= j && i < N);
            // lower triangular (row-major, store from column 0 to diagonal)
            return i * (i + 1) / 2 + j;
        }
    }
    static constexpr std::pair<int, int> inv_index(int idx) {
        assert(idx >= 0 && idx < StorageSize);
        if constexpr (TriangularType_ == Upper) {
            int i = 0;
            int offset = 0;
            while (i < N) {
                int row_len = N - i;
                if (idx < offset + row_len) break;
                offset += row_len;
                ++i;
            }
            int j = i + (idx - offset);
            return {i, j};
        } else {
            int i = 0;
            int offset = 0;
            while (i < N) {
                int row_len = i + 1;
                if (idx < offset + row_len) break;
                offset += row_len;
                ++i;
            }
            int j = idx - offset;
            return {i, j};
        }
    }
};


// TriangularMatrix = TriangularMatrixView + data ownership
template <typename Scalar_, int N_, int TriangularType_, bool NestAsRefBit_ = true>
class TriangularMatrix : public SquareMatrixBase<N_, TriangularMatrix<Scalar_, N_, TriangularType_, NestAsRefBit_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = TriangularMatrixView<Scalar_, N_, TriangularType_>;
    using MatrixType = TriangularMatrix<Scalar_, N_, TriangularType_, NestAsRefBit_>;
    using Base = SquareMatrixBase<N_, MatrixType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ + 1) / 2;
    static constexpr bool NestAsRefBit = NestAsRefBit_;
    static constexpr int TriangularType = TriangularType_;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | ((TriangularType == Upper) ? int(matrix_flags::upper_triangular) : int(matrix_flags::lower_triangular));


    // default constructor
    constexpr TriangularMatrix() : data_(), m_(data_.data()) { };

    // copy constructor
    constexpr TriangularMatrix(const MatrixType& other) : data_(), m_(data_.data()) { m_ = other; }

    // copy operator
    constexpr MatrixType& operator=(const MatrixType& other) { m_ = other; return *this; }

    // constructor from std::array
    constexpr explicit TriangularMatrix(const std::array<Scalar, StorageSize>& arr) : TriangularMatrix() { m_ = arr; }

    // constructor from C-style array
    constexpr explicit TriangularMatrix(const Scalar_ (&arr)[StorageSize]) : TriangularMatrix() { m_ = arr; }

    // constructor from std::vector
    constexpr explicit TriangularMatrix(const std::vector<Scalar>& vec) : TriangularMatrix() { m_ = vec; }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit TriangularMatrix(Callable callable) : TriangularMatrix() { m_ = callable; }

    // copy constructor from any MatrixBase-derived expression (templated)
    template<int OtherRows_, int OtherCols_, typename OtherDerived>
    constexpr explicit TriangularMatrix(const MatrixBase<OtherRows_,OtherCols_,OtherDerived>& xpr) : TriangularMatrix() { m_ = xpr; }

    // conversion constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template<typename Derived>
        explicit TriangularMatrix(const Eigen::MatrixBase<Derived>& other) : TriangularMatrix() { m_ = other; }
    #endif


    // static named constructors
    static constexpr TriangularMatrix Constant(Scalar c) {
        std::array<Scalar, StorageSize> data{};
        for (auto& val : data) val = c;
        return TriangularMatrix(data);
    }
    static constexpr TriangularMatrix Zero() { return Constant(Scalar(0)); }
    static constexpr TriangularMatrix Ones() { return Constant(Scalar(1)); }
    static constexpr TriangularMatrix NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // assignment from std::array
    constexpr TriangularMatrix& operator=(const std::array<Scalar, StorageSize>& rhs) { m_ = rhs; return *this; }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr TriangularMatrix& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) { m_ = rhs; return *this; }

    // assignment from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        TriangularMatrix& operator=(const Eigen::MatrixBase<Derived>& rhs) { m_ = rhs; return *this; }
    #endif

    // const access
    constexpr Scalar operator()(int i, int j) const { return m_(i, j); }
    constexpr Scalar operator[](int i) const requires(Rows == 1 || Cols == 1) { return m_[i]; }
    // non-const access
    constexpr Scalar& operator()(int i, int j) { return m_(i, j); }
    constexpr Scalar& operator[](int i) requires(Rows == 1 || Cols == 1)  { return m_[i]; }

    // convert to full matrix
    constexpr Matrix<Scalar, N, N> as_matrix() const {
        Matrix<Scalar, N, N> M;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                M(i, j) = MatrixViewType::in_triangle(i, j) ? (*this)(i, j) : Scalar(0);
        return M;
    }

    // convert to EigenMap
    #ifdef __FDAPDE_HAS_EIGEN__
        // TODO: differently from Matrix here I can not return a map because the Map saves the pointer to data_.data() but triangular matrices stores data in a non compatible way
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
    TriangularMatrixView<Scalar, N, TriangularType_> m_;

};

}

#endif // __FDAPDE_TRIANGULAR_MATRIX_H__