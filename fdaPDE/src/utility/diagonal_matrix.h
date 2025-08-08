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

#ifndef __FDAPDE_DIAGONAL_MATRIX_H__
#define __FDAPDE_DIAGONAL_MATRIX_H__

#include "header_check.h"
#include "matrix_base.h"
#include "square_matrix_base.h"
#include "diagonal_matrix.h"

namespace fdapde {

// forward declaration
template <typename Scalar_, int N> class DiagonalMatrixView;

// has_identity trait
namespace internals {

// DiagonalMatrix => has identity
template <typename Scalar_, int N, bool NestAsRefBit>
struct has_identity<DiagonalMatrix<Scalar_, N, NestAsRefBit>> : std::true_type {};

}

// is_view trait
namespace internals {

template <typename Scalar, int N>
struct is_view<DiagonalMatrixView<Scalar, N>> : std::true_type {};

}

// is_symmetric trait
namespace internals {

template <typename Scalar, int N>
struct is_symmetric<DiagonalMatrixView<Scalar, N>> : std::true_type {};
template <typename Scalar, int N, bool NestAsRefBit>
struct is_symmetric<DiagonalMatrix<Scalar, N, NestAsRefBit>> : std::true_type {};

}


// is_diagonal trait
namespace internals {

template <typename Scalar, int N>
struct is_diagonal<DiagonalMatrixView<Scalar, N>> : std::true_type {};
template <typename Scalar, int N, bool NestAsRefBit>
struct is_diagonal<DiagonalMatrix<Scalar, N, NestAsRefBit>> : std::true_type {};

}

// diagonal matrix view
template <typename Scalar_, int N_>
class DiagonalMatrixView : public SquareMatrixBase<N_, DiagonalMatrixView<Scalar_, N_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = DiagonalMatrixView<Scalar_, N_>;
    using Base = SquareMatrixBase<N_, MatrixViewType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_;
    static constexpr bool NestAsRefBit = true;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::diagonal) | int(matrix_flags::symmetric) | int(matrix_flags::upper_triangular) | int(matrix_flags::lower_triangular);

    // constructors
    constexpr DiagonalMatrixView() = delete;
    constexpr explicit DiagonalMatrixView(Scalar* ptr_data) : ptr_data_(ptr_data) {}
    constexpr explicit DiagonalMatrixView(std::array<Scalar, StorageSize>& data) : ptr_data_(data.data()) {}
    constexpr DiagonalMatrixView(const MatrixViewType& other) : ptr_data_(other.ptr_data_) { }

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

    // assignment from vector-like types
    template <typename VectorLikeType>
    constexpr MatrixViewType& operator=(const VectorLikeType& rhs)
    requires(internals::is_vector_like_v<VectorLikeType>){
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
            ptr_data_[id] = rhs.derived()(i, j); // j == i
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
                ptr_data_[id] = rhs.derived()(i, j); // j == i
            }
            return *this;
        }
    #endif

    // const access
    constexpr Scalar operator()(const int i, const int j) const {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols);
        return (i == j) ? ptr_data_[index(i, j)] : Scalar(0);
    }
    constexpr Scalar operator[](const int i) const
    requires(Cols == 1 || Rows == 1) {
        fdapde_assert(i >= 0 && i < StorageSize);
        return ptr_data_[i];
    }
    // non-const access
    /*
    constexpr Scalar& operator()(const int i, const int j) {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols && i == j);
        return ptr_data_[index(i, j)];
    }
    */
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
        assert(j == i && j < N);
        // only diagonal is stored
        return i;
    }
    static constexpr std::pair<int, int> inv_index(int idx) {
        assert(idx >= 0 && idx < StorageSize);
        return {idx, idx};
    }
};


// DiagonalMatrix = DiagonalMatrixView + data ownership
template <typename Scalar_, int N_, bool NestAsRefBit_ = true>
class DiagonalMatrix : public SquareMatrixBase<N_, DiagonalMatrix<Scalar_, N_, NestAsRefBit_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = DiagonalMatrixView<Scalar_, N_>;
    using MatrixType = DiagonalMatrix<Scalar_, N_, NestAsRefBit_>;
    using Base = SquareMatrixBase<N_, MatrixType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_;
    static constexpr bool NestAsRefBit = NestAsRefBit_;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::diagonal) | int(matrix_flags::symmetric) | int(matrix_flags::upper_triangular) | int(matrix_flags::lower_triangular);

    // default constructor
    constexpr DiagonalMatrix() : data_(), m_(data_.data()) { };

    // copy constructor
    constexpr DiagonalMatrix(const MatrixType& other) : data_(), m_(data_.data()) { m_ = other; }

    // copy operator
    constexpr MatrixType& operator=(const MatrixType& other) { m_ = other; return *this; }

    // constructor from std::array
    constexpr explicit DiagonalMatrix(const std::array<Scalar, StorageSize>& arr) : DiagonalMatrix() { m_ = arr; }

    // constructor from C-style array
    constexpr explicit DiagonalMatrix(const Scalar_ (&arr)[StorageSize]) : DiagonalMatrix() { m_ = arr; }

    // constructor from vector-like types
    template <typename VectorLikeType>
    constexpr explicit DiagonalMatrix(const VectorLikeType& vec)
    requires(internals::is_vector_like_v<VectorLikeType>) : DiagonalMatrix() {m_ = vec; }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit DiagonalMatrix(Callable callable)
    requires(std::is_invocable_v<Callable>) : DiagonalMatrix() { m_ = callable; }

    // copy constructor from any MatrixBase-derived expression (templated)
    template<int OtherRows_, int OtherCols_, typename OtherDerived>
    constexpr explicit DiagonalMatrix(const MatrixBase<OtherRows_,OtherCols_,OtherDerived>& xpr) : DiagonalMatrix() { m_ = xpr; }

    // conversion constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template<typename Derived>
        explicit DiagonalMatrix(const Eigen::MatrixBase<Derived>& other) : DiagonalMatrix() { m_ = other; }
    #endif


    // static named constructors
    static constexpr DiagonalMatrix Constant(Scalar c) {
        std::array<Scalar, StorageSize> data{};
        for (auto& val : data) val = c;
        return DiagonalMatrix(data);
    }
    static constexpr DiagonalMatrix Zero() { return Constant(Scalar(0)); }
    static constexpr DiagonalMatrix Ones() { return Constant(Scalar(1)); }
    static constexpr DiagonalMatrix NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // assignment from std::array
    constexpr DiagonalMatrix& operator=(const std::array<Scalar, StorageSize>& rhs) { m_ = rhs; return *this; }

    // constructor from vector-like types
    template <typename VectorLikeType>
    constexpr DiagonalMatrix& operator=(const VectorLikeType& rhs)
    requires(internals::is_vector_like_v<VectorLikeType>) { m_ = rhs; return *this; }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr DiagonalMatrix& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) { m_ = rhs; return *this; }

    // assignment from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        DiagonalMatrix& operator=(const Eigen::MatrixBase<Derived>& rhs) { m_ = rhs; return *this; }
    #endif

    // const access
    constexpr Scalar operator()(int i, int j) const { return m_(i, j); }
    constexpr Scalar operator[](int i) const requires(Rows == 1 || Cols == 1) { return m_[i]; }
    // non-const access
    // constexpr Scalar& operator()(int i, int j) { return m_(i, j); }
    constexpr Scalar& operator[](int i) requires(Rows == 1 || Cols == 1)  { return m_[i]; }

    // convert to full matrix
    constexpr Matrix<Scalar, N, N> as_matrix() const {
        Matrix<Scalar, N, N> M;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                M(i, j) = (i == j) ? (*this)(i, j) : Scalar(0);
        return M;
    }

    // convert to EigenMap
    #ifdef __FDAPDE_HAS_EIGEN__
        // TODO: differently from Matrix here I can not return a map because the Map saves the pointer to data_.data() but diagonal matrices stores data in a non compatible way
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
    DiagonalMatrixView<Scalar, N> m_;

};

}

#endif // __FDAPDE_DIAGONAL_MATRIX_H__