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

#ifndef __FDAPDE_SQUARE_MATRIX_BASE_H__
#define __FDAPDE_SQUARE_MATRIX_BASE_H__

#include "header_check.h"
#include "matrix_base.h"

namespace fdapde {

// forward declaration to break circular dependency
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, bool NestAsRefBit_> class Matrix;
template <typename Scalar_, int N_, bool NestAsRefBit_> class SymmetricMatrix;
template <typename Scalar_, int N_, bool NestAsRefBit_> class SkewSymmetricMatrix;
template <typename Derived, int ViewMode> struct TriangularView;
template <typename Derived> struct DiagonalView;
template <typename Derived> struct SymmetricPartView;
template <typename Derived> struct SkewSymmetricPartView;
template <int N, typename Derived> struct SquareMatrixBase;

// has_identity trait
namespace internals {

// default: no identity unless specialized
template <typename T>
struct has_identity : std::false_type {};

// helper variable template
template <typename T>
static constexpr bool has_identity_v = has_identity<T>::value;

}

// is_view trait
namespace internals {

template <typename Derived, int ViewMode>
struct is_view<TriangularView<Derived, ViewMode>> : std::true_type {};
template <typename Derived>
struct is_view<DiagonalView<Derived>> : std::true_type {};
template <typename Derived>
struct is_view<SymmetricPartView<Derived>> : std::true_type {};
template <typename Derived>
struct is_view<SkewSymmetricPartView<Derived>> : std::true_type {};

}


[[maybe_unused]] constexpr int Upper = 0;       // lower triangular view of matrix
[[maybe_unused]] constexpr int Lower = 1;       // upper triangular view of matrix
[[maybe_unused]] constexpr int UnitUpper = 2;   // lower triangular view of matrix with ones on the diagonal
[[maybe_unused]] constexpr int UnitLower = 3;   // upper triangular view of matrix with ones on the diagonal

// is_triangular and directional‐triangular traits
namespace internals {

// default: nothing is triangular
template <typename T>
struct is_triangular : std::false_type {};

// default: not upper‐triangular
template <typename T>
struct is_upper_triangular : std::false_type {};

// default: not lower‐triangular
template <typename T>
struct is_lower_triangular : std::false_type {};

// helper variable templates
template <typename T>
static constexpr bool is_triangular_v = is_triangular<T>::value;

template <typename T>
static constexpr bool is_upper_triangular_v = is_upper_triangular<T>::value;

template <typename T>
static constexpr bool is_lower_triangular_v = is_lower_triangular<T>::value;

// Specialize for any TriangularView
template <typename Derived, int ViewMode>
struct is_triangular<TriangularView<Derived, ViewMode>> : std::true_type {};

// Upper‐type views: both strict and unit
template <typename Derived>
struct is_upper_triangular<TriangularView<Derived, Upper>> : std::true_type {};
template <typename Derived>
struct is_upper_triangular<TriangularView<Derived, UnitUpper>> : std::true_type {};

// Lower‐type views: both strict and unit
template <typename Derived>
struct is_lower_triangular<TriangularView<Derived, Lower>> : std::true_type {};
template <typename Derived>
struct is_lower_triangular<TriangularView<Derived, UnitLower>> : std::true_type {};

} // namespace internals


// triangular view
template <typename Derived, int ViewMode_>
struct TriangularView : public SquareMatrixBase<Derived::Rows, TriangularView<Derived, ViewMode_>> {

    fdapde_static_assert(
      Derived::Rows != 1 && Derived::Cols != 1 && Derived::Rows == Derived::Cols,
      TRIANGULAR_VIEW_DEFINED_ONLY_FOR_SQUARED_MATRICES);

    using Base = SquareMatrixBase<Derived::Rows, TriangularView<Derived, ViewMode_>>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Cols = Derived::Rows;
    static constexpr int Rows = Derived::Cols;
    static constexpr bool NestAsRefBit = false;
    static constexpr bool ReadOnly = Derived::ReadOnly;
    static constexpr int XprBits = Derived::XprBits;
    static constexpr int ViewMode = ViewMode_;

    // constructors
    constexpr TriangularView() = default;
    constexpr explicit TriangularView(const Derived& xpr) : xpr_(xpr) { }

    // const access
    constexpr Scalar operator()(int i, int j) const {
        fdapde_static_assert(ViewMode >= 0 && ViewMode <=4, VIEW_MODE_IN_UNKNOWN);
        if constexpr (ViewMode == Upper) return i > j ? 0 : xpr_(i, j);
        if constexpr (ViewMode == Lower) return i < j ? 0 : xpr_(i, j);
        if constexpr (ViewMode == UnitUpper) return i > j ? 0 : (i == j ? 1 : xpr_(i, j));
        if constexpr (ViewMode == UnitLower) return i < j ? 0 : (i == j ? 1 : xpr_(i, j));
        return std::numeric_limits<Scalar>::quiet_NaN();
    }

    // block assignment
    template <int Rows_, int Cols_, typename RhsType>
    constexpr TriangularView<Derived, ViewMode>& operator=(const MatrixBase<Rows_, Cols_, RhsType>& rhs) {
        fdapde_static_assert(Derived::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          ViewMode == Upper || ViewMode == Lower, TRIANGULAR_BLOCK_ASSIGNMENT_REQUIRES_EITHER_UPPER_OR_LOWER_VIEW);
        fdapde_static_assert(
          Derived::Rows == Rows_ && Derived::Cols_ == Cols &&
            std::is_convertible_v<typename RhsType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_SIZE_OR_YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        int row = 0, col = 0;
        for (int i = 0; i < Rows_; ++i) {
            for (int j = 0; j < i; ++j) {
                if constexpr (ViewMode == Lower) { row = i; col = j; }
                if constexpr (ViewMode == Upper) { row = j; col = i; }
                xpr_(row, col) = rhs(row, col);
            }
        }
	    // assign diagonal
        for (int i = 0; i < Rows_; ++i) { xpr_(i, i) = rhs(i, i); }
        return *this;
    }

   private:
    internals::ref_select_t<const Derived> xpr_;
};

// diagonal view
template <typename Derived>
struct DiagonalView : public SquareMatrixBase<Derived::Rows, DiagonalView<Derived>> {

    fdapde_static_assert(Derived::Rows == Derived::Cols, DIAGONAL_BLOCK_DEFINED_ONLY_FOR_SQUARED_MATRICES);

    using Base = SquareMatrixBase<Derived::Rows, DiagonalView<Derived>>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr bool NestAsRefBit = false;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = Derived::XprBits;


    // constructors
    constexpr DiagonalView() = default;
    constexpr explicit DiagonalView(Derived& xpr) : xpr_(xpr) { }

    // const access
    constexpr Scalar operator()(int i, int j) const { return i == j ? xpr_(i, i) : 0; }
    constexpr const Scalar& operator[](int i) const { return xpr_(i, i); }
    // non-const access
    constexpr Scalar& operator[](int i) { return xpr_(i, i); }

    // assignment operator
    template <typename RhsType> constexpr DiagonalView<Derived>& operator=(const RhsType& rhs) {
        fdapde_static_assert(Derived::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          RhsType::Cols == 1 && RhsType::Rows == Rows &&
            std::is_convertible_v<typename RhsType::Scalar FDAPDE_COMMA Scalar>,
          VECTOR_REQUIRED_OR_YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int i = 0; i < Rows; ++i) { xpr_(i, i) = rhs[i]; }
        return *this;
    }

private:
    internals::ref_select_t<Derived> xpr_;
};

// is_symmetric trait
namespace internals {

template <typename Derived>
struct is_symmetric<SymmetricPartView<Derived>> : std::true_type {};

}

// symmetric view
template <typename Derived>
struct SymmetricPartView : public SquareMatrixBase<Derived::Rows, SymmetricPartView<Derived>> {
    fdapde_static_assert(Derived::Rows == Derived::Cols, SYMMETRIC_VIEW_DEFINED_ONLY_FOR_SQUARE_MATRICES);

    using Scalar = typename Derived::Scalar;
    static constexpr int N = Derived::Rows;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr bool NestAsRefBit = false;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::symmetric);

    // constructors
    constexpr SymmetricPartView() = default;
    constexpr explicit SymmetricPartView(const Derived& xpr) : xpr_(xpr) {}

    // element access
    constexpr Scalar operator()(int i, int j) const {
        return 0.5 * (xpr_(i, j) + xpr_(j, i));
    }

private:
    internals::ref_select_t<Derived> xpr_;
};

// skew-symmetric view
template <typename Derived>
struct SkewSymmetricPartView : public SquareMatrixBase<Derived::Rows, SkewSymmetricPartView<Derived>> {
    fdapde_static_assert(Derived::Rows == Derived::Cols, SKEW_SYMMETRIC_VIEW_DEFINED_ONLY_FOR_SQUARE_MATRICES);

    using Scalar = typename Derived::Scalar;
    static constexpr int N = Derived::Rows;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr bool NestAsRefBit = false;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::skew_symmetric);

    // constructors
    constexpr SkewSymmetricPartView() = default;
    constexpr explicit SkewSymmetricPartView(const Derived& xpr) : xpr_(xpr) {}

    // element access
    constexpr Scalar operator()(int i, int j) const {
        return 0.5 * (xpr_(i, j) - xpr_(j, i));
    }

private:
    internals::ref_select_t<Derived> xpr_;
};

template <int N, typename Derived>
struct SquareMatrixBase : public MatrixBase<N, N, Derived> {
    using Base = MatrixBase<N, N, Derived>;

    // Base methods
    using Base::derived;

    // triangular view of matrix expression
    template <int ViewMode>
    constexpr TriangularView<const Derived, ViewMode> triangular_view() const {
        return TriangularView<const Derived, ViewMode>(derived());
    }
    template <int ViewMode>
    constexpr TriangularView<Derived, ViewMode> triangular_view() {
        return TriangularView<Derived, ViewMode>(derived());
    }

    // diagonal view of matrix expression
    constexpr DiagonalView<const Derived> diagonal() const { return DiagonalView<const Derived>(derived()); }
    constexpr DiagonalView<Derived> diagonal() { return DiagonalView<Derived>(derived()); }

    // symmetric part view of matrix expression
    constexpr SymmetricPartView<const Derived> symmetric_part() const {
        return SymmetricPartView<Derived>(derived());
    }
    constexpr SymmetricPartView<Derived> symmetric_part() {
        return SymmetricPartView<Derived>(derived());
    }

    // skew-symmetric part view of matrix expression
    constexpr SkewSymmetricPartView<const Derived> skew_symmetric_part() const {
        return SkewSymmetricPartView<Derived>(derived());
    }
    constexpr SkewSymmetricPartView<Derived> skew_symmetric_part() {
        return SkewSymmetricPartView<Derived>(derived());
    }

    // trace of matrix
    constexpr auto trace() const {
        typename Derived::Scalar trace_ = 0;
        for (int i = 0; i < N; ++i) trace_ += derived().operator()(i, i);
        return trace_;
    }

    // off diagonal L^2 norm
    constexpr auto off_diagonal_norm() const { return fdapde::sqrt(Base::squared_norm() - diagonal().squared_norm()); }

    // is symmetric check
    [[nodiscard]] constexpr bool is_symmetric(double tol = 1e-12) const {
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                if (std::fabs(derived().operator()(i, j) - derived().operator()(j, i)) > tol)
                    return false;
            }
        }
        return true;
    }

    // is skew-symmetric check
    [[nodiscard]] constexpr bool is_skew_symmetric(double tol = 1e-12) const {
        for (int i = 0; i < N; ++i) {
            if (std::fabs(derived().operator()(i, i)) > tol) return false;
            for (int j = i + 1; j < N; ++j) {
                if (std::fabs(derived().operator()(i, j) + derived().operator()(j, i)) > tol) return false;
            }
        }
        return true;
    }

    // static named constructor (Identity matrix)
    // TODO: it would be optimal to return a SPDMatrix once they exist
    static constexpr auto Identity() requires(internals::has_identity_v<Derived>) {
        using SM = SymmetricMatrix<typename Derived::Scalar, N, Derived::NestAsRefBit>;
        SM I(SM::Zero());
        for (int i = 0; i < N; ++i)
                I(i, i) = typename Derived::Scalar(1);
        return I;
    }

    // TODO: Eigenvalue Decomposition

};

}

#endif //__FDAPDE_SQUARE_MATRIX_BASE_H__