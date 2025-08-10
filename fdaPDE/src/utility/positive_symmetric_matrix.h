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

#ifndef __FDAPDE_POSITIVE_SYMMETRIC_MATRIX_H__
#define __FDAPDE_POSITIVE_SYMMETRIC_MATRIX_H__

#include "header_check.h"
#include "matrix_base.h"
#include "square_matrix_base.h"
#include "matrix_decomposition.h"

namespace fdapde {

// forward declaration
template <typename Scalar, int N, bool StrictPD_> class PositiveSymmetricMatrixView;
template <typename Scalar, int N, bool StrictPD_, bool NestAsRefBit> class PositiveSymmetricMatrix;

// convenient aliases
template <typename Scalar, int N> using SPDMatrixView  = PositiveSymmetricMatrixView<Scalar, N, true>;
template <typename Scalar, int N> using SPSDMatrixView = PositiveSymmetricMatrixView<Scalar, N, false>;
template <typename Scalar, int N, bool NestAsRefBit = true>
using SPDMatrix  = PositiveSymmetricMatrix<Scalar, N, true,  NestAsRefBit>;
template <typename Scalar, int N, bool NestAsRefBit = true>
using SPSDMatrix = PositiveSymmetricMatrix<Scalar, N, false, NestAsRefBit>;

// has_identity trait
namespace internals {

// PositiveSymmetricMatrix => has identity
template <typename Scalar, int N, bool StrictPD, bool NestAsRefBit>
struct has_identity<PositiveSymmetricMatrix<Scalar, N, StrictPD, NestAsRefBit>> : std::true_type {};

}

// is_view trait
namespace internals {

template <typename Scalar, int N, bool StrictPD>
struct is_view<PositiveSymmetricMatrixView<Scalar, N, StrictPD>> : std::true_type {};

}

// is_symmetric trait
namespace internals {

template <typename Scalar, int N, bool StrictPD>
struct is_symmetric<PositiveSymmetricMatrixView<Scalar, N, StrictPD>> : std::true_type {};
template <typename Scalar, int N, bool StrictPD, bool NestAsRefBit>
struct is_symmetric<PositiveSymmetricMatrix<Scalar, N, StrictPD, NestAsRefBit>> : std::true_type {};

}

// PositiveSymmetricMatrixView (StrictPD=true -> SPD, StrictPD=false -> SPSD)
template <typename Scalar_, int N_, bool StrictPD_>
class PositiveSymmetricMatrixView : public SquareMatrixBase<N_, PositiveSymmetricMatrixView<Scalar_, N_, StrictPD_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = PositiveSymmetricMatrixView<Scalar_, N_, StrictPD_>;
    using Base = SquareMatrixBase<N_, MatrixViewType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ + 1) / 2;
    static constexpr int StorageOrder = RowMajor;
    static constexpr bool NestAsRefBit = true;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits =
        int(matrix_flags::square) |
        int(matrix_flags::symmetric) |
        (StrictPD_ ? int(matrix_flags::spd) : int(matrix_flags::spsd));

    // constructors
    constexpr PositiveSymmetricMatrixView() = delete;
    constexpr explicit PositiveSymmetricMatrixView(Scalar* ptr_data) : ptr_data_(ptr_data) { } // impossible to have a default initializer here
    constexpr explicit PositiveSymmetricMatrixView(std::array<Scalar, StorageSize>& data) : ptr_data_(data.data()) { check(); }
    constexpr PositiveSymmetricMatrixView(const MatrixViewType& other) : ptr_data_(other.ptr_data_) { }
    // NOTE: as in OrthogonalMatrixView, we avoid calling check() in copy ctor to prevent loops

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

    // assignment from C-array
    constexpr MatrixViewType& operator=(const Scalar (&rhs)[StorageSize]) {
        for (int id = 0; id < StorageSize; ++id) {
            ptr_data_[id] = rhs[id];
        }
        check();
        return *this;
    }

    // assignment from std::vector
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
          std::is_convertible_v<typename decltype(std::function { callable })::result_type FDAPDE_COMMA std::array<Scalar FDAPDE_COMMA StorageSize>>,
          CALLABLE_DOES_NOT_RETURN_SOMETHING_CONVERTIBLE_TO_AN_ARRAY_OF_SCALAR);
        *this = callable();
        check();
        return *this;
    }

    // assignment from MatrixBase expression (symmetrized)
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr MatrixViewType& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_DIMENSIONS_OR_YOU_ARE_TRYING_TO_ASSIGN_A_RHS_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int id = 0; id < StorageSize; ++id) {
            auto [i, j] = inv_index(id);
            ptr_data_[id] = Scalar(0.5) * (rhs.derived()(i, j) + rhs.derived()(j, i));
        }
        check();
        return *this;
    }

    // assignment from Eigen matrix (symmetrized)
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename OtherDerived>
        MatrixViewType& operator=(const Eigen::MatrixBase<OtherDerived>& rhs) {
            fdapde_static_assert(
              OtherDerived::RowsAtCompileTime != Dynamic && OtherDerived::ColsAtCompileTime != Dynamic &&
                std::is_convertible_v<typename OtherDerived::Scalar FDAPDE_COMMA Scalar>,
              CANNOT_ASSIGN_FROM_EIGEN_HEAP_ALLOCATED_MATRIX_OR_INVALID_SCALAR_TYPE);
            for (int id = 0; id < StorageSize; ++id) {
                auto [i, j] = inv_index(id);
                ptr_data_[id] = Scalar(0.5) * (rhs.derived()(i, j) + rhs.derived()(j, i));
            }
            check();
            return *this;
        }
    #endif

    // const access
    constexpr Scalar operator()(const int i, const int j) const {
        fdapde_assert(i >= 0 && i < Rows && j >= 0 && j < Cols);
        return (j >= i) ? ptr_data_[index(i, j)] : ptr_data_[index(j, i)];
    }

    // data
    constexpr const Scalar_* data() const { return ptr_data_; }
    constexpr Scalar_* data() { return ptr_data_; }

    // eigen-decomposition getters (forward evd_)
    constexpr const auto& eigenvalues() const { return evd_.eigenvalues(); }
    constexpr const auto& eigenvectors() const { return evd_.eigenvectors(); }

    // check using eigenvalue decomposition
    void check() {
        std::cout << *this << std::endl;
        evd_.compute(*this);
        const auto& evals = evd_.eigenvalues();
        bool ok = true;
        for (int k = 0; k < N; ++k) {
            if constexpr (StrictPD_) {
                ok = ok && (evals[k] > Scalar(0));
            } else {
                ok = ok && (evals[k] >= Scalar(0));
            }
        }
        fdapde_constexpr_assert(ok) // The matrix provided is not (semi-)positive definite
    }

protected:
    Scalar_* ptr_data_ = nullptr;
    EigenDecomposition<MatrixViewType> evd_;

    // indexes (upper-triangular packed, column-major blocks)
    static constexpr int index(const int i, const int j){
        assert(j >= i && j < N);
        // assumes j >= i
        return i * (2 * N - i + 1) / 2 + (j - i);
    }
    static constexpr std::pair<int, int> inv_index(const int idx) {
        assert(idx >= 0 && idx < StorageSize);
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
    }
};


// PositiveSymmetricMatrix = PositiveSymmetricMatrixView + data ownership
template <typename Scalar_, int N_, bool StrictPD_, bool NestAsRefBit_ = true>
class PositiveSymmetricMatrix : public SquareMatrixBase<N_, PositiveSymmetricMatrix<Scalar_, N_, StrictPD_, NestAsRefBit_>> {
    fdapde_static_assert(N_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);

public:
    using MatrixViewType = PositiveSymmetricMatrixView<Scalar_, N_, StrictPD_>;
    using MatrixType = PositiveSymmetricMatrix<Scalar_, N_, StrictPD_, NestAsRefBit_>;
    using Base = SquareMatrixBase<N_, MatrixType>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int StorageSize = N_ * (N_ + 1) / 2;
    static constexpr int StorageOrder = RowMajor;
    static constexpr bool NestAsRefBit = NestAsRefBit_;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits =
        int(matrix_flags::square) |
        int(matrix_flags::symmetric) |
        (StrictPD_ ? int(matrix_flags::spd) : int(matrix_flags::spsd));

    // default constructor -> Identity (valid SPD/SPSD)
    constexpr PositiveSymmetricMatrix() : data_(), m_(data_.data()) {
        m_ = DiagonalMatrix<Scalar, N>::Identity();
        m_.check();
    };

    // copy constructor
    constexpr PositiveSymmetricMatrix(const MatrixType& other) : data_(), m_(data_.data()) {
        m_ = other;
        m_.check();
    }

    // copy operator
    constexpr MatrixType& operator=(const MatrixType& other) { m_ = other; return *this; }

    // constructor from std::array
    constexpr explicit PositiveSymmetricMatrix(const std::array<Scalar, StorageSize>& arr) : PositiveSymmetricMatrix() { m_ = arr; }

    // constructor from C-style array
    constexpr explicit PositiveSymmetricMatrix(const Scalar_ (&arr)[StorageSize]) : PositiveSymmetricMatrix() { m_ = arr; }

    // constructor from std::vector
    constexpr explicit PositiveSymmetricMatrix(const std::vector<Scalar>& vec) : PositiveSymmetricMatrix() { m_ = vec; }

    // constructor from callable returning array<Scalar, StorageSize>
    template <typename Callable>
    constexpr explicit PositiveSymmetricMatrix(Callable callable) : PositiveSymmetricMatrix() { m_ = callable; }

    // copy constructor from any MatrixBase-derived expression (templated)
    template<int OtherRows_, int OtherCols_, typename OtherDerived>
    constexpr explicit PositiveSymmetricMatrix(const MatrixBase<OtherRows_, OtherCols_, OtherDerived>& xpr) : PositiveSymmetricMatrix() { m_ = xpr; }

    // conversion constructor from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template<typename Derived>
        explicit PositiveSymmetricMatrix(const Eigen::MatrixBase<Derived>& other) : PositiveSymmetricMatrix() { m_ = other; }
    #endif

    // assignment from std::array
    constexpr MatrixType& operator=(const std::array<Scalar, StorageSize>& rhs) { m_ = rhs; return *this; }

    // assignment from MatrixBase expression
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr MatrixType& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) { m_ = rhs; return *this; }

    // assignment from Eigen matrix
    #ifdef __FDAPDE_HAS_EIGEN__
        template <typename Derived>
        MatrixType& operator=(const Eigen::MatrixBase<Derived>& rhs) { m_ = rhs; return *this; }
    #endif

    // const access
    constexpr Scalar operator()(int i, int j) const { return m_(i, j); }

    // eigen-decomposition getters (forward evd_)
    constexpr const auto& eigenvalues() const { return m_.eigenvalues(); }
    constexpr const auto& eigenvectors() const { return m_.eigenvectors(); }

    // data
    constexpr const Scalar* data() const { return m_.data(); }
    Scalar* data() { return m_.data(); }
    constexpr const std::array<Scalar,StorageSize>& storage() const { return data_; }

private:
    std::array<Scalar, StorageSize> data_;
    PositiveSymmetricMatrixView<Scalar, N, StrictPD_> m_;
};

}

#endif   // __FDAPDE_POSITIVE_SYMMETRIC_MATRIX_H__