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

#ifndef __FDAPDE_ORTHOGONAL_MATRIX_H__
#define __FDAPDE_ORTHOGONAL_MATRIX_H__

#include "header_check.h"
#include "diagonal_matrix.h"

namespace fdapde {

// OrthogonalMatrixView
template <typename Scalar_, int N_>
class OrthogonalMatrixView : public SquareMatrixBase<N_, OrthogonalMatrixView<Scalar_, N_>> {
public:
    using MatrixViewType = OrthogonalMatrixView<Scalar_, N_>;
    using Base = SquareMatrixBase<N_, MatrixViewType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * N_;
    static constexpr bool NestAsRefBit = true;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::orthogonal);

    // constructors
    constexpr OrthogonalMatrixView() = delete;
    constexpr explicit OrthogonalMatrixView(Scalar* ptr_data) : ptr_data_(ptr_data) { } // impossible to have a default initializer of OrthogonalMatrix
    constexpr explicit OrthogonalMatrixView(std::array<Scalar, StorageSize>& data) : ptr_data_(data.data()) { check(); }
    constexpr OrthogonalMatrixView(const MatrixViewType& other) : ptr_data_(other.ptr_data_) { } // infinite loop -> segfault
    // TODO: check if there is a way to add the check() to these two constructors as well (motivations are in the comments)

    // copy operator
    constexpr MatrixViewType& operator=(const MatrixViewType& other) {
        if (this == &other) return *this;
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = other.data()[id];
        }
        check();
        return *this;
    }

    // assignment from std::array
    constexpr MatrixViewType& operator=(const std::array<Scalar, StorageSize>& rhs) {
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = rhs[id];
        }
        check();
        return *this;
    }

    // assignment from std::array
    constexpr MatrixViewType& operator=(const Scalar (&rhs)[StorageSize]) {
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = rhs[id];
        }
        check();
        return *this;
    }

    // assignment from C-array
    constexpr MatrixViewType& operator=(const std::vector<Scalar>& rhs) {
        fdapde_constexpr_assert(rhs.size() == StorageSize);
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = rhs[id];
        }
        check();
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
        check();
        return *this;
    }

    // assignment from MatrixBase expression
    template <int RhsN, typename RhsXprType>
    constexpr MatrixViewType& operator=(const SquareMatrixBase<RhsN, RhsXprType>& rhs) {
        fdapde_static_assert(
          RhsN == N &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_DIMENSIONS_OR_YOU_ARE_TRYING_TO_ASSIGN_A_RHS_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int id = 0; id < StorageSize; ++id) {
            auto[i, j] = inv_index(id);
            ptr_data_[id] = rhs.derived()(i, j);
        }
        check();
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
            check();
            return *this;
        }
    #endif

    // const access
    constexpr Scalar operator()(const int i, const int j) const {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols);
        return ptr_data_[index(i, j)];
    }

    // convert to EigenMap
    #ifdef __FDAPDE_HAS_EIGEN__
        auto as_eigen_map() {
            return Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Eigen::ColMajor>>(ptr_data_);
        }
    #endif

    // data
    constexpr const Scalar_* data() const { return ptr_data_; }
    constexpr Scalar_* data() { return ptr_data_; }

    // check
    void check() {
        bool orthogonality = almost_equal(this->transpose() * (*this), DiagonalMatrix<Scalar, N>::Identity());
        fdapde_constexpr_assert(orthogonality) // The matrix provided is not an orthogonal matrix
    }

protected:
    Scalar_* ptr_data_ = nullptr;

    // indexes
    static constexpr int index(const int i, const int j) {
        return i + j * N;
    }
    static constexpr std::pair<int, int> inv_index(const int id) {
        int i = id % N;
        int j = id / N;
        return {i, j};
    }
};


// OrthogonalMatrix = OrthogonalMatrixView + data ownership
template <typename Scalar_, int N_, bool NestAsRefBit_ = true>
class OrthogonalMatrix : public SquareMatrixBase<N_, OrthogonalMatrix<Scalar_, N_, NestAsRefBit_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = OrthogonalMatrixView<Scalar_, N_>;
    using MatrixType = OrthogonalMatrix<Scalar_, N_, NestAsRefBit_>;
    using Base = SquareMatrixBase<N_, MatrixType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * N_;
    static constexpr bool NestAsRefBit = NestAsRefBit_;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::orthogonal);

    // default constructor
    constexpr OrthogonalMatrix() : data_(), m_(data_.data()) {
        m_ = OrthogonalMatrix::Identity();
        m_.check();
    };

    // copy constructor
    constexpr OrthogonalMatrix(const MatrixType& other) : data_(), m_(data_.data()) { m_ = other; }

    // copy operator
    constexpr MatrixType& operator=(const MatrixType& other) { m_ = other; return *this; }

    // constructor from std::array
    constexpr explicit OrthogonalMatrix(const std::array<Scalar, StorageSize>& arr) : OrthogonalMatrix() { m_ = arr; }

    // constructor from C-style array
    constexpr explicit OrthogonalMatrix(const Scalar_ (&arr)[StorageSize]) : OrthogonalMatrix() { m_ = arr; }

    // constructor from std::vector
    constexpr explicit OrthogonalMatrix(const std::vector<Scalar>& vec) : OrthogonalMatrix() { m_ = vec; }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit OrthogonalMatrix(Callable callable) : OrthogonalMatrix() { m_ = callable; }

    // copy constructor from any MatrixBase-derived expression (templated)
    template<typename Derived>
    constexpr explicit OrthogonalMatrix(const MatrixBase<N, N, Derived>& xpr) : OrthogonalMatrix() { m_ = xpr; }

    // conversion constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template<typename Derived>
        explicit OrthogonalMatrix(const Eigen::MatrixBase<Derived>& other) : OrthogonalMatrix() { m_ = other; }
    #endif

    // assignment from std::array
    constexpr OrthogonalMatrix& operator=(const std::array<Scalar, StorageSize>& rhs) { m_ = rhs; return *this; }

    // assignment from MatrixBase expression
    template <int RhsN_, typename RhsXprType>
    constexpr OrthogonalMatrix& operator=(const SquareMatrixBase<RhsN_, RhsXprType>& rhs) { m_ = rhs; return *this; }

    // assignment from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        OrthogonalMatrix& operator=(const Eigen::MatrixBase<Derived>& rhs) { m_ = rhs; return *this; }
    #endif

    // const access
    constexpr Scalar operator()(int i, int j) const { return m_(i, j); }

    // data
    constexpr const Scalar* data() const { return m_.data(); }
    Scalar* data() { return m_.data(); }
    constexpr const std::array<Scalar,StorageSize>& storage() const { return data_; }

private:
    std::array<Scalar, StorageSize> data_;
    OrthogonalMatrixView<Scalar, N> m_;

};


}

#endif   // _FDAPDE_ORTHOGONAL_MATRIX_H__
