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
// forward declaration
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, bool ReadOnly_ > class MatrixView;
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, bool NestAsRefBit_> class Matrix;

// is_view trait
namespace internals {

template <typename Scalar, int Rows, int Cols, int StorageOrder, bool ReadOnly>
struct is_view<MatrixView<Scalar, Rows, Cols, StorageOrder, ReadOnly>> : std::true_type {};

}

// has_identity trait
namespace internals {

// Matrix<Scalar, N, N> => has identity
template <typename Scalar_, int N_, int StorageOrder_, bool NestAsRefBit_>
struct has_identity<Matrix<Scalar_, N_, N_, StorageOrder_, NestAsRefBit_>> : std::true_type {};

}

// maps an existing array of data to a cexpr::Matrix. This can be used also to integrate Eigen with cexpr linear algebra
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor, bool ReadOnly_ = false>
class MatrixView :
public std::conditional_t<Rows_== Cols_,
    SquareMatrixBase<Rows_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_, ReadOnly_>>,
    MatrixBase<Rows_, Cols_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_, ReadOnly_>>
> {
    fdapde_static_assert(Rows_ > 0 && Cols_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = MatrixView<Scalar_, Rows_, Cols_, StorageOrder_, ReadOnly_>;
    using Base = std::conditional_t<Rows_==Cols_, SquareMatrixBase<Rows_, MatrixViewType>, MatrixBase<Rows_, Cols_, MatrixViewType>>;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageSize = Rows_ * Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRefBit = false;
    static constexpr bool ReadOnly = ReadOnly_;
    static constexpr int XprBits = (Rows_==Cols_) ? int(matrix_flags::square) : int(matrix_flags::none);

    // ptr type
    using PtrType = std::conditional_t<ReadOnly, const Scalar*, Scalar*>;

    // default strides for a dense layout
    static constexpr int DefaultOuterStride = (StorageOrder == RowMajor ? Cols : Rows);
    static constexpr int DefaultInnerStride = 1;

    // constructors
    constexpr MatrixView() = delete;
    constexpr explicit MatrixView(Scalar* ptr_data) : ptr_data_(ptr_data) {}
    constexpr explicit MatrixView(std::array<Scalar, StorageSize>& data) : ptr_data_(data.data()) {}
    constexpr MatrixView(const MatrixViewType& other) : ptr_data_(other.ptr_data_) { }
    template <bool NestAsRef>
    constexpr explicit MatrixView(const Matrix<Scalar, Rows, Cols, StorageOrder, NestAsRef>& other) : ptr_data_(other.data()) { }

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

    // assignment from std::array
    constexpr MatrixViewType& operator=(const Scalar (&rhs)[StorageSize]) {
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = rhs[id];
        }
        return *this;
    }

    // assignment from C-array
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
        return ptr_data_[index(i, j)];
    }
    constexpr Scalar operator[](const int i) const
    requires(Cols == 1 || Rows == 1) {
        fdapde_assert(i >= 0 && i < StorageSize);
        return ptr_data_[i];
    }
    // non-const access
    constexpr Scalar& operator()(const int i, const int j)
    requires (!ReadOnly) {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols);
        return ptr_data_[index(i, j)];
    }
    constexpr Scalar& operator[](const int i)
    requires(!ReadOnly && (Cols == 1 || Rows == 1)) {
        fdapde_assert(i >= 0 && i < StorageSize);
        return ptr_data_[i];
    }

    // convert to EigenMap
    #ifdef __FDAPDE_HAS_EIGEN__
        auto as_eigen_map() requires(StorageOrder == ColMajor) {
            return Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Eigen::ColMajor>>(ptr_data_);
        }
        auto as_eigen_map() requires(StorageOrder == RowMajor) {
            return Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor>>(ptr_data_);
        }
    #endif

    // setters
    constexpr void setConstant(Scalar c) {
        for (int id = 0; id < StorageSize; ++id) ptr_data_[id] = c;
    }
    constexpr void setZero() { setConstant(Scalar(0)); }
    constexpr void setOnes() { setConstant(Scalar(1)); }

    // dimensions
    static constexpr int rows() { return Rows; }
    static constexpr int cols() { return Cols; }
    static constexpr int innerStride() { return inner_stride_; }
    static constexpr int outerStride() { return outer_stride_; }
    static constexpr int rowStride() { return StorageOrder_ == RowMajor ? outerStride() : innerStride(); }
    static constexpr int colStride() { return StorageOrder_ == RowMajor ? innerStride() : outerStride(); }

    // data
    constexpr const Scalar_* data() const { return ptr_data_; }
    constexpr Scalar_* data() { return ptr_data_; }

protected:
    PtrType ptr_data_ = nullptr;
    static constexpr int outer_stride_ = DefaultOuterStride;   // increment between two consecutive rows (RowMajor) or columns (ColMajor)
    static constexpr int inner_stride_ = DefaultInnerStride;   // increment between two consecutive entries within a row (RowMajor) or column (ColMajor)

    // indexes
    static constexpr int index(const int i, const int j) {
        return i * rowStride() + j * colStride();
    }
    static constexpr std::pair<int, int> inv_index(const int id) {
        int i, j;
        if (StorageOrder == RowMajor) {
            i = id / Cols;
            j = id % Cols;
        } else {
            i = id % Rows;
            j = id / Rows;
        }
        return {i, j};
    }
};


// Matrix = MatrixView + data ownership
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor, bool NestAsRefBit_ = true>
class Matrix :
public std::conditional_t<Rows_ == Cols_,
    SquareMatrixBase<Rows_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_, NestAsRefBit_>>,
    MatrixBase<Rows_, Cols_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_, NestAsRefBit_>>
> {
    fdapde_static_assert(Rows_ > 0 && Cols_ > 0, EMPTY_MATRIX_IS_ILL_FORMED);

public:
    using MatrixViewType = MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>;
    using MatrixType = Matrix<Scalar_, Rows_, Cols_, StorageOrder_, NestAsRefBit_>;
    using Base = std::conditional_t<Rows_ == Cols_, SquareMatrixBase<Rows_, MatrixType>, MatrixBase<Rows_, Cols_, MatrixType>>;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageSize = Rows_ * Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr bool NestAsRefBit = NestAsRefBit_;
    static constexpr bool ReadOnly = false;
    static constexpr int XprBits = (Rows_ == Cols_) ? int(matrix_flags::square) : int(matrix_flags::none);

    // default constructor
    constexpr Matrix() : data_(), m_(data_.data()) { };

    // copy constructor
    constexpr Matrix(const MatrixType& other) : data_(), m_(data_.data()) { m_ = other; }

    // copy operator
    constexpr MatrixType& operator=(const MatrixType& other) { m_ = other; return *this; }

    // constructor from std::array
    constexpr explicit Matrix(const std::array<Scalar, StorageSize>& arr) : Matrix() { m_ = arr; }

    // constructor from C-style array
    constexpr explicit Matrix(const Scalar_ (&arr)[StorageSize]) : Matrix() { m_ = arr; }

    // constructor from std::vector
    constexpr explicit Matrix(const std::vector<Scalar>& vec) : Matrix() { m_ = vec; }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit Matrix(Callable callable) : Matrix() { m_ = callable; }

    // copy constructor from any MatrixBase-derived expression (templated)
    template<typename Derived>
    constexpr explicit Matrix(const MatrixBase<Rows,Cols,Derived>& xpr) : Matrix() { m_ = xpr; }

    // conversion constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template<typename Derived>
        explicit Matrix(const Eigen::MatrixBase<Derived>& other) : Matrix() { m_ = other; }
    #endif

    // scalar constructor for 1D vector/matrix (StorageSize == 1)
    constexpr explicit Matrix(Scalar x) : Matrix() {
        fdapde_static_assert(StorageSize == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_ONE_ELEMENT);
        data_[0] = x;
    }

    // constructor for 2D vector (StorageSize == 2)
    constexpr explicit Matrix(Scalar x, Scalar y) : Matrix() {
        fdapde_static_assert(StorageSize == 2, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_TWO_ELEMENTS);
        data_ = {x, y};
    }

    // constructor for 3D vector (StorageSize == 3)
    constexpr explicit Matrix(Scalar x, Scalar y, Scalar z) : Matrix() {
        fdapde_static_assert(StorageSize == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_THREE_ELEMENTS);
        data_ = {x, y, z};
    }

    // static named constructors
    static constexpr Matrix Constant(Scalar c) {
        std::array<Scalar, StorageSize> data{};
        for (auto& val : data) val = c;
        return Matrix(data);
    }
    static constexpr Matrix Zero() { return Constant(Scalar(0)); }
    static constexpr Matrix Ones() { return Constant(Scalar(1)); }
    static constexpr Matrix NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }

    // assignment from std::array
    constexpr Matrix& operator=(const std::array<Scalar, StorageSize>& rhs) { m_ = rhs; return *this; }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr Matrix& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) { m_ = rhs; return *this; }

    // assignment from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        Matrix& operator=(const Eigen::MatrixBase<Derived>& rhs) { m_ = rhs; return *this; }
    #endif

    // const access
    constexpr Scalar operator()(int i, int j) const { return m_(i, j); }
    constexpr Scalar operator[](int i) const requires(Rows == 1 || Cols == 1) { return m_[i]; }
    // non-const access
    constexpr Scalar& operator()(int i, int j) { return m_(i, j); }
    constexpr Scalar& operator[](int i) requires(Rows == 1 || Cols == 1)  { return m_[i]; }

    // convert to EigenMap
    #ifdef __FDAPDE_HAS_EIGEN__
        auto as_eigen_map() { return m_.as_eigen_map(); }
    #endif

    // setters
    constexpr void setConstant(Scalar c) { m_.setConstant(c); }
    constexpr void setZero() { setConstant(Scalar(0)); }
    constexpr void setOnes() { setConstant(Scalar(1)); }

    // dimensions
    static constexpr int rows() { return Rows; }
    static constexpr int cols() { return Cols; }

    // data
    constexpr const Scalar* data() const { return m_.data(); }
    Scalar* data() { return m_.data(); }
    constexpr const std::array<Scalar,StorageSize>& storage() const { return data_; }

   private:
    std::array<Scalar, StorageSize> data_;
    MatrixView<Scalar, Rows, Cols, StorageOrder> m_;

};


}


#endif   // _FDAPDE_MATRIX_H__