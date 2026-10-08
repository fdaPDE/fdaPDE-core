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

#ifndef __FDAPDE_LINALG_SYMMETRIC_H__
#define __FDAPDE_LINALG_SYMMETRIC_H__

#include "header_check.h"
#include "symmetric_cache.h"

namespace fdapde {

// symmetric matrix type system
/// @brief provides symmetric matrix expressions and eigendecomposition access
template <typename XprType> struct SymmetricMatrixExpr;
/// @brief views packed lower-triangular storage as a symmetric matrix
template <typename Scalar_, int Order_, typename Policy_ = Cache::None, int StorageOrder_ = RowMajor>
class SymmetricMatrixView;
/// @brief owns packed symmetric coefficients with an optional spectral cache
template <typename Scalar_, int Order_, typename Policy_ = Cache::None, int StorageOrder_ = RowMajor>
class SymmetricMatrix;

namespace internals {

/// @brief declares detection of matrix sources that can prepare spectral quantities
template <typename T> struct is_spectral_cache_source;

/// @brief identifies a view whose scalar type permits writes
template <typename Scalar, int Order, typename Policy, int StorageOrder>
struct is_mutable_matrix_view<SymmetricMatrixView<Scalar, Order, Policy, StorageOrder>> :
    std::bool_constant<!std::is_const_v<Scalar>> { };

}   // namespace internals

// forward decls
/// @brief computes the eigendecomposition of a symmetric matrix
template <typename XprType> class EVD;

namespace internals {

// class wrapping a generic expression to the expression of a symmetric matrix. internal usage only
/// @brief mirrors one selected triangle of a square expression
template <int ViewMode_, typename SymmetricXprType_>
struct symmetric_wrapper : public SymmetricMatrixExpr<symmetric_wrapper<ViewMode_, SymmetricXprType_>> {
   private:
    fdapde_static_assert(ViewMode_ == Lower || ViewMode_ == Upper, VIEW_MODE_MUST_BE_EITHER_LOWER_OR_UPPER);
    using Base = SymmetricMatrixExpr<symmetric_wrapper<ViewMode_, SymmetricXprType_>>;
    using XprType = std::remove_reference_t<SymmetricXprType_>;
    using XprTypeClean = std::remove_cv_t<XprType>;
    using XprTypeNested = internals::ref_select_t<SymmetricXprType_>;
    static constexpr int ViewMode = ViewMode_;
   public:
    using Scalar = typename XprTypeClean::Scalar;
    static constexpr int Rows = XprTypeClean::Rows;
    static constexpr int Cols = XprTypeClean::Cols;
    static constexpr int StorageOrder = XprTypeClean::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief copies the symmetric adaptor while retaining its nested expression
    constexpr symmetric_wrapper(const symmetric_wrapper&) = default;
    /// @brief borrows one triangle of a nonempty square expression and mirrors it across the diagonal
    template <typename XprType__>
        requires(!std::same_as<std::remove_cvref_t<XprType__>, symmetric_wrapper> &&
                 internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr symmetric_wrapper(XprType__&& xpr) :
        Base(), xpr_(std::forward<XprType__>(xpr)), size_(xpr_.rows() == xpr_.cols() ? xpr_.rows() : 0) {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
        fdapde_assert(
          !(xpr_.rows() <= 0 || xpr_.rows() != xpr_.cols()), std::invalid_argument,
          "symmetric view requires positive square dimensions");
    }
    /// @brief reflects the selected source triangle across the main diagonal
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, size_, size_);
        if constexpr (ViewMode == Upper) { return i > j ? xpr_(j, i) : xpr_(i, j); }
        if constexpr (ViewMode == Lower) { return i < j ? xpr_(j, i) : xpr_(i, j); }
    }
    /// @brief returns the row count
    constexpr int rows() const { return size_; }
    /// @brief returns the column count
    constexpr int cols() const { return size_; }
    /// @brief returns the stored representation
    constexpr const XprTypeNested& rep() const { return xpr_; }
   private:
    XprTypeNested xpr_;
    int size_;
};

// helper cast function
/// @brief adapts an expression to symmetric matrix operations
template <int ViewMode_, typename XprType_> constexpr auto symmetric_cast(XprType_&& xpr) {
    return symmetric_wrapper<ViewMode_, XprType_>(std::forward<XprType_>(xpr));
}

}   // namespace internals

/// @brief provides symmetric matrix expressions and eigendecomposition access
template <typename XprType_> struct SymmetricMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using Base = MatrixExpr<XprType_>;
    using Base::derived;
    // inherit assignment from base
    using Base::operator=;

    /// @brief computes the symmetric eigendecomposition
    auto evd() const { return EVD<XprType>(derived()); }

    /// @brief returns an independent eigenvalue vector, reusing a native spectral cache when available
    /// @details preserves scalar precision and matrix order; eigenvalues are not guaranteed to be sorted
    /// a reusable cache copies only eigenvalues; otherwise the native eigensolver supplies the spectrum
    /// controlled writes refresh the cache on demand, while mutable data() exposure forces every request to refresh
    /// the returned coefficients remain valid after source mutation or destruction, including temporary sources
    auto eigenvalues() const {
        using Scalar = std::remove_cv_t<typename XprType::Scalar>;
        using Values = Vector<Scalar, XprType::Rows>;
        if constexpr (internals::is_spectral_cache_source<XprType>::value) {
            return Values(derived().cache().eigenvalues());
        } else {
            const auto decomposition = evd();
            return Values(decomposition.eigenvalues());
        }
    }

    /// @brief returns an owning orthogonal eigenvector basis, reusing a native spectral cache when available
    /// @details columns correspond to eigenvalues() for unchanged input; sorting and column signs are unspecified
    /// repeated eigenvalues do not select a canonical basis for their eigenspace
    /// tracked writes invalidate factors for on-demand refresh; mutable data() exposure forces every request to refresh
    /// the returned matrix remains valid after source mutation or destruction, including temporary sources
    auto eigenvectors() const {
        using Scalar = std::remove_cv_t<typename XprType::Scalar>;
        using Vectors = OrthogonalMatrix<Scalar, XprType::Rows, XprType::Rows>;
        if constexpr (internals::is_spectral_cache_source<XprType>::value) {
            return Vectors(derived().cache().eigenvectors(), unchecked);
        } else {
            const auto decomposition = evd();
            return Vectors(decomposition.eigenvectors());
        }
    }
    /// @brief returns an owning symmetric inverse using cached eigenpairs or a pivoted dense solve
    template <typename Policy = Cache::None>
    auto inv() const
        requires std::floating_point<std::remove_cv_t<typename XprType::Scalar>>;
    /// @brief returns the checked SPD exponential with the requested output cache policy
    /// @details reuses input eigenpairs when available and certifies the rounded reconstructed output
    template <typename Policy = Cache::None> auto exp() const;

    // internal triangular matrix representation
    /// @brief returns the stored representation
    constexpr decltype(auto) rep() const { return derived().rep(); }
    /// @brief returns the stored representation
    constexpr decltype(auto) rep() { return derived().rep(); }
};

/// @brief maps symmetric coordinates to packed coefficients and invalidates caches on writes
template <typename Scalar_, int Order_, typename Policy_, int StorageOrder_, typename SymmetricMatrixType>
class SymmetricMatrixBase : public SymmetricMatrixExpr<SymmetricMatrixType> {
   private:
    using Base = SymmetricMatrixExpr<SymmetricMatrixType>;
    using Base::derived;
   public:
    using Scalar = std::remove_const_t<Scalar_>;
    using CachePolicy = Policy_;
    using CacheSlot = internals::symmetric_cache_slot<Scalar, Order_, Policy_>;
    static constexpr int Rows = Order_, Cols = Order_, StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    static constexpr int ViewMode = Lower;
    using assignment_executor = internals::triangular_assignment_executor;
    using Base::operator=;

    /// @brief retains a coefficient alias whose later writes invalidate the shared cache slot
    class coefficient_proxy {
       public:
        /// @brief binds one packed coefficient to its optional spectral cache
        constexpr coefficient_proxy(Scalar* value, CacheSlot* cache) : value_(value), cache_(cache) { }
        /// @brief copies a saved coefficient and its shared invalidation binding
        constexpr coefficient_proxy(const coefficient_proxy&) = default;
        /// @brief reads the current coefficient without invalidating cached quantities
        constexpr operator Scalar() const { return *value_; }
        /// @brief stores a coefficient and invalidates any selected cached quantities
        constexpr coefficient_proxy& operator=(Scalar value) {
            if constexpr (Policy_::Flags) {
                // spectral decompositions require finite coefficients even when supplied through saved aliases
                fdapde_strong_assert(
                  std::isfinite(value), std::invalid_argument, "symmetric coefficients must be finite");
                cache_->invalidate();
            }
            *value_ = value;
            return *this;
        }
        /// @brief copies the referenced coefficient without rebinding a saved alias
        constexpr coefficient_proxy& operator=(const coefficient_proxy& other) { return *this = Scalar(other); }
        /// @brief adds to the coefficient through the same invalidation path as assignment
        constexpr coefficient_proxy& operator+=(Scalar value) { return *this = Scalar(*this) + value; }
        /// @brief subtracts from the coefficient through the same invalidation path as assignment
        constexpr coefficient_proxy& operator-=(Scalar value) { return *this = Scalar(*this) - value; }
        /// @brief scales the coefficient through the same invalidation path as assignment
        constexpr coefficient_proxy& operator*=(Scalar value) { return *this = Scalar(*this) * value; }
        /// @brief divides the coefficient through the same invalidation path as assignment
        constexpr coefficient_proxy& operator/=(Scalar value) { return *this = Scalar(*this) / value; }
       private:
        Scalar* value_;
        CacheSlot* cache_;
    };
    using reference = coefficient_proxy;
    using const_reference = Scalar;

    /// @brief initializes the common symmetric expression interface
    constexpr SymmetricMatrixBase() = default;
    /// @brief reads a coefficient shared by its two mirrored coordinates
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, derived().rows(), derived().cols());
        if (i < j) std::swap(i, j);
        return derived().coefficient_data_()[std::size_t(i) * (i + 1) / 2 + j];
    }
    /// @brief binds a writable coefficient without exposing an untracked pointer
    constexpr auto operator()(int i, int j)
        requires(ReadOnly == 0)
    {
        internals::validate_matrix_index(i, j, derived().rows(), derived().cols());
        if (i < j) std::swap(i, j);
        return reference(derived().coefficient_data_() + std::size_t(i) * (i + 1) / 2 + j, derived().cache_slot_());
    }
};

// symmetric matrices vector-space structure (additive group)
/// @brief adds equally shaped operands while preserving symmetry
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(const SymmetricMatrixExpr<LhsXprType>& lhs, const SymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::symmetric_cast<Lower>(
      static_cast<const MatrixExpr<LhsXprType>&>(lhs) + static_cast<const MatrixExpr<RhsXprType>&>(rhs));
}
/// @brief subtracts equally shaped operands while preserving symmetry
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(const SymmetricMatrixExpr<LhsXprType>& lhs, const SymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::symmetric_cast<Lower>(
      static_cast<const MatrixExpr<LhsXprType>&>(lhs) - static_cast<const MatrixExpr<RhsXprType>&>(rhs));
}
/// @brief scales by the right scalar while preserving symmetry
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const SymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::symmetric_cast<Lower>(static_cast<const MatrixExpr<XprType>&>(lhs) * rhs);
}
/// @brief scales by the left scalar while preserving symmetry
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(ScalarType lhs, const SymmetricMatrixExpr<XprType>& rhs) {
    return internals::symmetric_cast<Lower>(lhs * static_cast<const MatrixExpr<XprType>&>(rhs));
}
/// @brief divides each coefficient by the scalar while preserving symmetry
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const SymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::symmetric_cast<Lower>(static_cast<const MatrixExpr<XprType>&>(lhs) / rhs);
}
// any other operation doesn't preserve symmetry. A raw MatrixExpr is returned

namespace internals {
/// @brief validates packed lower-triangular coordinate counts before allocation or coefficient reads
template <int Order> constexpr int packed_symmetric_order(std::size_t count) {
    fdapde_strong_assert(count > 0, std::invalid_argument, "symmetric coordinates: vector must be nonempty");
    fdapde_strong_assert(
      count <= static_cast<std::size_t>(std::numeric_limits<int>::max()), std::length_error,
      "symmetric coordinates: packed storage size exceeds supported range");
    const int order = compute_triangular_shape(static_cast<int>(count));
    fdapde_strong_assert(order > 0, std::invalid_argument, "symmetric coordinates: count must be triangular");
    fdapde_strong_assert(
      Order == Dynamic || Order == order, std::invalid_argument,
      "symmetric coordinates: count does not match the matrix dimensions");
    return order;
}
}   // namespace internals

/// @brief owns packed symmetric coefficients with optional lazily refreshed eigenpairs
/// @details mutable raw data access permanently disables cache reuse for the current storage binding
template <typename Scalar_, int Order_, typename Policy_, int StorageOrder_>
class SymmetricMatrix :
    public SymmetricMatrixBase<
      Scalar_, Order_, Policy_, StorageOrder_, SymmetricMatrix<Scalar_, Order_, Policy_, StorageOrder_>> {
   private:
    using Base = SymmetricMatrixBase<Scalar_, Order_, Policy_, StorageOrder_, SymmetricMatrix>;
    using StorageType = TriangularMatrix<Scalar_, Order_, Order_, Lower, StorageOrder_>;
   public:
    // packed storage supports positive fixed orders and arithmetic scalars; spectral work requires floating point
    fdapde_static_assert(Order_ == Dynamic || Order_ > 0, INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    static_assert(std::same_as<Scalar_, std::remove_cv_t<Scalar_>>, "symmetric owner scalar must be unqualified");
    static_assert(
      Policy_::Flags == 0 || std::is_floating_point_v<Scalar_>, "symmetric spectral cache requires floating point");
    using Scalar = Scalar_;
    using CachePolicy = Policy_;
    using CacheSlot = typename Base::CacheSlot;
    using View = SymmetricMatrixView<Scalar_, Order_, Policy_, StorageOrder_>;
    using ConstView = SymmetricMatrixView<const Scalar_, Order_, Policy_, StorageOrder_>;
    static constexpr int NestAsRef = 1;
    using Base::Cols;
    using Base::Rows;

    /// @brief value-initializes fixed coefficients and leaves dynamic storage empty
    constexpr SymmetricMatrix() : data_() { initialize_cache_(); }
    /// @brief copies coefficients and reusable eigenpairs into independent storage
    constexpr SymmetricMatrix(const SymmetricMatrix& other) : data_(other.data_) {
        if constexpr (Policy_::Flags) cache_ = std::make_unique<CacheOwner>(*other.cache_);
    }
    /// @brief copies expiring coefficients without invalidating the source's existing views
    constexpr SymmetricMatrix(SymmetricMatrix&& other) : SymmetricMatrix(static_cast<const SymmetricMatrix&>(other)) { }
    /// @brief copies independent source storage while preserving same-shape destination view bindings
    constexpr SymmetricMatrix& operator=(const SymmetricMatrix& other) & {
        if (this != &other) commit_(other);
        return *this;
    }
    /// @brief rejects replacement of a temporary owner
    constexpr void operator=(const SymmetricMatrix&) && = delete;
    /// @brief allocates dynamic zero coefficients of the requested order
    constexpr explicit SymmetricMatrix(int order) : SymmetricMatrix(order, order) { }
    /// @brief accepts equal dimensions from generic matrix allocation code
    constexpr SymmetricMatrix(int rows, int cols) : data_(rows, cols) { initialize_cache_(); }
    /// @brief copies a symmetric expression into independent packed coefficients
    template <typename Xpr>
    constexpr SymmetricMatrix(const SymmetricMatrixExpr<Xpr>& source) :
        SymmetricMatrix(static_cast<const MatrixExpr<Xpr>&>(source)) { }
    /// @brief materializes a matrix operation or copies a square matrix's symmetric lower triangle
    template <typename Xpr>
        requires(!internals::is_vector_shaped_v<Xpr> || requires(const Xpr& value) { value.eval_matrix(); })
    constexpr explicit SymmetricMatrix(const MatrixExpr<Xpr>& source) : data_(checked_storage_(source)) {
        initialize_cache_();
    }
    /// @brief copies packed native coordinates in lower-triangle row order
    template <typename Xpr>
        requires(internals::is_vector_shaped_v<Xpr> && !requires(const Xpr& value) { value.eval_matrix(); })
    constexpr explicit SymmetricMatrix(const MatrixExpr<Xpr>& coordinates) : data_() {
        const auto& values = coordinates.derived();
        const int order = internals::packed_symmetric_order<Rows>(values.size());
        if constexpr (Rows == Dynamic) data_.resize(order, order);
        for (int k = 0; k < values.size(); ++k) data_.data()[k] = static_cast<Scalar>(values[k]);
        validate_coefficients_();
        initialize_cache_();
    }
    /// @brief copies packed vector coordinates and infers the dynamic order from their count
    template <typename S>
        requires(std::is_constructible_v<Scalar, S>)
    constexpr explicit SymmetricMatrix(const std::vector<S>& coordinates) : data_(coordinates) {
        validate_coefficients_();
        initialize_cache_();
    }
    /// @brief copies a C array of fixed packed coordinates
    template <typename S, std::size_t Size>
        requires(std::is_constructible_v<Scalar, S>)
    constexpr explicit SymmetricMatrix(const S (&coordinates)[Size]) : data_(coordinates) {
        validate_coefficients_();
        initialize_cache_();
    }
    /// @brief copies array coordinates in packed lower-triangle order
    template <typename S, std::size_t Size>
        requires(std::is_constructible_v<Scalar, S>)
    constexpr explicit SymmetricMatrix(const std::array<S, Size>& coordinates) : data_() {
        const int order = internals::packed_symmetric_order<Rows>(Size);
        if constexpr (Rows == Dynamic) data_.resize(order, order);
        std::copy(coordinates.begin(), coordinates.end(), data_.data());
        validate_coefficients_();
        initialize_cache_();
    }
    /// @brief validates and materializes a source before replacing coefficients and cache state
    template <typename Xpr> constexpr SymmetricMatrix& operator=(const MatrixExpr<Xpr>& source) & {
        SymmetricMatrix candidate(source);
        commit_(candidate);
        return *this;
    }
    /// @brief materializes coefficientwise values before replacing the owner's coefficients
    template <typename Xpr> constexpr SymmetricMatrix& operator=(const MatrixCoeffWiseExpr<Xpr>& source) & {
        return *this = source.mwise();
    }
    /// @brief changes the order of dynamic packed storage and discards obsolete eigenpairs
    void resize(int order) { resize(order, order); }
    /// @brief accepts equal dimensions from generic dynamic matrix resizing code
    void resize(int rows, int cols) {
        const int old_order = this->rows();
        data_.resize(rows, cols);
        if constexpr (Policy_::Flags) {
            if (old_order != this->rows()) initialize_cache_();
        }
    }
    /// @brief returns the matrix order
    constexpr int rows() const { return data_.rows(); }
    /// @brief returns the matrix order
    constexpr int cols() const { return data_.cols(); }
    /// @brief borrows the complete symmetric coefficients without permitting writes
    constexpr ConstView rep() const& { return ConstView(*this); }
    /// @brief binds mutable coefficients to the owner's invalidation slot
    constexpr View rep() & { return view(); }
    /// @brief rejects a representation borrowing an expiring owner
    void rep() const&& = delete;
    /// @brief exposes read-only packed coefficients without changing cache reuse
    /// @details call std::as_const(matrix).data() to select this overload on a mutable owner
    /// this access neither invalidates a ready cache nor restores reuse disabled by earlier mutable data access
    constexpr const Scalar* data() const { return data_.data(); }
    /// @brief exposes writable packed coefficients and permanently disables cache reuse for this binding
    /// @details the call itself invalidates prepared eigenpairs, even if no coefficient is subsequently changed
    /// a retained pointer may write after any cache refresh, so every later cache() or evd() request recomputes
    /// this restriction survives same-shape assignment and is shared by all views of the owner
    /// assigning the result to const Scalar* still selects this mutable overload; use std::as_const(matrix).data()
    /// for reads, or coefficient proxies and view assignment for writes that preserve normal cache reuse
    /// request cache() again after raw writes; previously borrowed slots and factor views cannot detect them
    /// independent owning copies have their own reusable cache; Cache::None has no cache state to invalidate
    /// the pointer borrows packed lower-triangle storage and becomes invalid on owner destruction or shape change
    constexpr Scalar* data() {
        if constexpr (Policy_::Flags) cache_->slot.expose_mutable_data();
        return data_.data();
    }
    /// @brief borrows mutable coefficients and their cache invalidation slot
    constexpr View view() & { return View(*this); }
    /// @brief borrows read-only coefficients and the shared spectral slot
    constexpr ConstView view() const& { return ConstView(*this); }
    /// @brief rejects a view borrowing an expiring owner
    void view() const&& = delete;
    /// @brief prepares eigenpairs after writes and on every call following mutable raw exposure
    /// @details after mutable data() access even repeated requests without intervening writes recompute eigenpairs
    /// the returned slot is borrowed; after a raw write request cache() again before reading spectral quantities
    const CacheSlot& cache() const&
        requires(Policy_::Flags != 0)
    {
        cache_->slot.prepare(internals::symmetric_cast<Lower>(data_));
        return cache_->slot;
    }
    /// @brief rejects a cache reference borrowing an expiring owner
    void cache() const&&
        requires(Policy_::Flags != 0)
    = delete;
   private:
    friend Base;
    template <typename, int, typename, int> friend class SymmetricMatrixView;
    using CacheOwner = internals::matrix_cache_owner<CacheSlot>;
    /// @brief binds optional cache storage to the current packed order
    constexpr void initialize_cache_() {
        if constexpr (Policy_::Flags) cache_ = std::make_unique<CacheOwner>(rows());
    }
    /// @brief rejects nonfinite packed coefficients when spectral quantities are requested
    constexpr void validate_coefficients_() const {
        if constexpr (Policy_::Flags) {
            // eigenpair caching is defined only for finite coefficients
            for (int k = 0; k < rows() * (rows() + 1) / 2; ++k)
                fdapde_strong_assert(
                  std::isfinite(data_.data()[k]), std::invalid_argument, "symmetric coefficients must be finite");
        }
    }
    /// @brief evaluates matrix-level operations once and validates cache-enabled input before packing
    template <typename Xpr> static constexpr StorageType checked_storage_(const MatrixExpr<Xpr>& source) {
        if constexpr (requires { source.derived().eval_matrix(); }) {
            return checked_storage_(source.derived().eval_matrix());
        } else {
            if constexpr (Policy_::Flags) {
                // square finite symmetric input is required before a spectral cache can represent the value
                fdapde_strong_assert(
                  source.rows() >= 0 && source.rows() == source.cols() && (Rows == Dynamic || source.rows() == Rows),
                  std::invalid_argument, "symmetric input has incompatible shape");
                const Matrix<Scalar, Rows, Cols> dense(source);
                for (int i = 0; i < dense.rows(); ++i)
                    for (int j = 0; j < dense.cols(); ++j) {
                        fdapde_strong_assert(
                          std::isfinite(dense(i, j)), std::invalid_argument, "symmetric coefficients must be finite");
                    }
                const Scalar tolerance =
                  Scalar(32) * dense.rows() * std::numeric_limits<Scalar>::epsilon() * dense.inf_norm();
                for (int i = 0; i < dense.rows(); ++i)
                    for (int j = 0; j < i; ++j)
                        fdapde_strong_assert(
                          std::abs(dense(i, j) - dense(j, i)) <= tolerance, std::invalid_argument,
                          "symmetric input must be symmetric");
                return StorageType(dense);
            } else
                return StorageType(source);
        }
    }
    /// @brief copies a candidate while retaining invalidation bindings when the matrix order is unchanged
    constexpr void commit_(const SymmetricMatrix& candidate) {
        const int old_order = rows();
        data_ = candidate.data_;
        if constexpr (Policy_::Flags) {
            if (old_order != rows()) initialize_cache_();
            cache_->slot.copy_from(candidate.cache_->slot);
        }
    }
    /// @brief provides trusted read access without altering cache reuse
    constexpr const Scalar* coefficient_data_() const { return data_.data(); }
    /// @brief provides proxy writes without exposing raw storage publicly
    constexpr Scalar* coefficient_data_() { return data_.data(); }
    /// @brief returns the shared invalidation slot or no slot for a disabled cache
    constexpr CacheSlot* cache_slot_() {
        if constexpr (Policy_::Flags)
            return &cache_->slot;
        else
            return nullptr;
    }
    StorageType data_;
    [[no_unique_address]] std::conditional_t<
      Policy_::Flags == 0, internals::empty_spd_cache<3>, std::unique_ptr<CacheOwner>> cache_;
};

/// @brief borrows packed symmetric coefficients and shares optional cache invalidation
/// @details owner shape changes or destruction invalidate the view and saved coefficient proxies
template <typename Scalar_, int Order_, typename Policy_, int StorageOrder_>
class SymmetricMatrixView :
    public SymmetricMatrixBase<
      Scalar_, Order_, Policy_, StorageOrder_, SymmetricMatrixView<Scalar_, Order_, Policy_, StorageOrder_>> {
   private:
    using Base = SymmetricMatrixBase<Scalar_, Order_, Policy_, StorageOrder_, SymmetricMatrixView>;
   public:
    // supported packed layouts and scalar policies match the owning symmetric matrix
    fdapde_static_assert(Order_ == Dynamic || Order_ > 0, INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    using Scalar = std::remove_const_t<Scalar_>;
    using Owner = SymmetricMatrix<Scalar, Order_, Policy_, StorageOrder_>;
    using CachePolicy = Policy_;
    using CacheSlot = typename Base::CacheSlot;
    using View = SymmetricMatrixView<Scalar, Order_, Policy_, StorageOrder_>;
    using ConstView = SymmetricMatrixView<const Scalar, Order_, Policy_, StorageOrder_>;
    static constexpr int NestAsRef = 0;
    using Base::Cols;
    using Base::ReadOnly;
    using Base::Rows;
    /// @brief copies the external coefficient and invalidation bindings
    constexpr SymmetricMatrixView(const SymmetricMatrixView&) = default;
    /// @brief creates an empty dynamic view without cache storage
    constexpr SymmetricMatrixView()
        requires(Order_ == Dynamic && Policy_::Flags == 0)
    = default;
    /// @brief borrows fixed packed storage without an external spectral slot
    constexpr explicit SymmetricMatrixView(Scalar_* data)
        requires(Policy_::Flags == 0)
        : SymmetricMatrixView(data, Order_, Order_) {
        // pointer-only construction must supply its shape through a positive compile-time order
        fdapde_static_assert(Order_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    /// @brief binds external packed storage and its dynamic or fixed order
    constexpr SymmetricMatrixView(Scalar_* data, int order)
        requires(Policy_::Flags == 0)
        : SymmetricMatrixView(data, order, order) { }
    /// @brief accepts equal dimensions from generic cache-free view construction
    constexpr SymmetricMatrixView(Scalar_* data, int rows, int cols)
        requires(Policy_::Flags == 0)
        : data_(data), n_(rows) {
        // reuse the packed triangle's shape, overflow and nonnull-storage contract
        (void)TriangularMatrixView<Scalar_, Order_, Order_, Lower, StorageOrder_>(data, rows, cols);
    }
    /// @brief borrows writable coefficients and their owner's cache slot
    constexpr explicit SymmetricMatrixView(Owner& owner) : data_(owner.coefficient_data_()), n_(owner.rows()) {
        if constexpr (Policy_::Flags) cache_ = owner.cache_slot_();
    }
    /// @brief borrows read-only coefficients and their owner's cache slot
    constexpr explicit SymmetricMatrixView(const Owner& owner)
        requires(std::is_const_v<Scalar_>)
        : data_(owner.data()), n_(owner.rows()) {
        if constexpr (Policy_::Flags) cache_ = &owner.cache_->slot;
    }
    /// @brief rejects a view into an expiring owner
    SymmetricMatrixView(Owner&&) = delete;
    /// @brief rejects a view into an expiring const owner
    SymmetricMatrixView(const Owner&&) = delete;
    /// @brief converts mutable coefficient access into a read-only binding
    constexpr SymmetricMatrixView(const View& other)
        requires(std::is_const_v<Scalar_>)
        : data_(other.data_), n_(other.n_), cache_(other.cache_) { }
    /// @brief returns the bound matrix order
    constexpr int rows() const { return n_; }
    /// @brief returns the bound matrix order
    constexpr int cols() const { return n_; }
    /// @brief retains read-only access to the complete symmetric coefficients and their cache slot
    constexpr ConstView rep() const { return ConstView(*this); }
    /// @brief retains controlled writes when borrowing the representation
    constexpr SymmetricMatrixView rep()
        requires(!std::is_const_v<Scalar_>)
    {
        return *this;
    }
    /// @brief exposes packed coefficients without permitting untracked mutation
    /// @details call std::as_const(view).data() for read-only access through a mutable view
    /// this preserves the shared cache state but cannot restore reuse disabled by earlier mutable data access
    constexpr const Scalar* data() const { return data_; }
    /// @brief exposes writable coefficients and permanently disables reuse through the shared slot
    /// @details the call immediately invalidates the owner's or batch element's eigenpairs, even without a write
    /// every later cache() or evd() request through any alias of that element recomputes its eigenpairs
    /// same-shape assignment and destruction of this view do not restore reuse; other batch slots are unaffected
    /// assigning the result to const Scalar* still calls this overload; select std::as_const(view).data() for reads
    /// prefer coefficient proxies or view assignment for writes that keep automatic cache reuse
    /// after raw writes request cache() again; borrowed slots and factor views cannot observe those writes
    /// Cache::None has no cache state to invalidate; the pointer remains subject to the underlying owner's lifetime
    constexpr Scalar* data()
        requires(!std::is_const_v<Scalar_>)
    {
        if constexpr (Policy_::Flags) cache_->expose_mutable_data();
        return data_;
    }
    /// @brief prepares eigenpairs against current packed coefficients before borrowing their slot
    /// @details mutable raw exposure through any alias disables reuse for this shared slot
    /// request cache() again after raw writes instead of reading a previously borrowed slot or factor view
    const CacheSlot& cache() const
        requires(Policy_::Flags != 0)
    {
        cache_->prepare(internals::symmetric_cast<Lower>(rep()));
        return *cache_;
    }
    /// @brief assigns a complete source snapshot without changing this view's binding
    template <typename Xpr>
    constexpr SymmetricMatrixView& assign(const MatrixExpr<Xpr>& source)
        requires(ReadOnly == 0)
    {
        // views cannot change shape because their coefficients and slot belong to another object
        fdapde_strong_assert(
          source.rows() == n_ && source.cols() == n_, std::invalid_argument,
          "symmetric view assignment cannot change shape");
        const Owner candidate(source);
        std::copy_n(candidate.data(), std::size_t(n_) * (n_ + 1) / 2, data_);
        if constexpr (Policy_::Flags) cache_->invalidate();
        return *this;
    }
    /// @brief assigns a source snapshot while retaining the current binding
    template <typename Xpr>
    constexpr SymmetricMatrixView& operator=(const MatrixExpr<Xpr>& source) &
        requires(ReadOnly == 0)
    {
        return assign(source);
    }
    /// @brief assigns a source snapshot and returns the temporary view binding by value
    template <typename Xpr>
      constexpr SymmetricMatrixView operator=(const MatrixExpr<Xpr>& source) &&
      requires(ReadOnly == 0) {
          assign(source);
          return *this;
      }
      /// @brief materializes coefficientwise values while retaining the view binding
      template <typename Xpr>
      constexpr SymmetricMatrixView& operator=(const MatrixCoeffWiseExpr<Xpr>& source) &
          requires(ReadOnly == 0)
    {
        return assign(source.mwise());
    }
    /// @brief assigns coefficientwise values and returns the temporary view binding by value
    template <typename Xpr>
      constexpr SymmetricMatrixView operator=(const MatrixCoeffWiseExpr<Xpr>& source) &&
      requires(ReadOnly == 0) {
          assign(source.mwise());
          return *this;
      }
      /// @brief copies coefficients without rebinding this mutable view
      constexpr SymmetricMatrixView& operator=(const SymmetricMatrixView& source) &
          requires(ReadOnly == 0)
    {
        return assign(source);
    }
    /// @brief copies coefficients and returns the temporary view binding by value
    constexpr SymmetricMatrixView operator=(const SymmetricMatrixView& source) &&
      requires(ReadOnly == 0) {
          assign(source);
          return *this;
      }
      /// @brief rejects replacement through a read-only coefficient binding
      void operator=(const SymmetricMatrixView&)
          requires(ReadOnly != 0)
      = delete;
   private:
    friend Base;
    template <typename, int> friend class MatrixBatch;
    template <typename, int, typename, int> friend class SymmetricMatrixView;
    /// @brief binds batch-owned packed coefficients to their optional shared invalidation slot
    constexpr SymmetricMatrixView(Scalar_* data, int order, CacheSlot* cache) : data_(data), n_(order) {
        if constexpr (Policy_::Flags) cache_ = cache;
    }
    /// @brief provides trusted read access without changing cache reuse
    constexpr const Scalar* coefficient_data_() const { return data_; }
    /// @brief provides controlled proxy access without exposing the raw binding
    constexpr Scalar_* coefficient_data_() { return data_; }
    /// @brief returns the shared invalidation slot when a spectral policy is active
    constexpr CacheSlot* cache_slot_() {
        if constexpr (Policy_::Flags)
            return cache_;
        else
            return nullptr;
    }
    Scalar_* data_ = nullptr;
    int n_ = Order_ == Dynamic ? 0 : Order_;
    [[no_unique_address]] std::conditional_t<Policy_::Flags == 0, internals::empty_spd_cache<4>, CacheSlot*> cache_ {};
};

namespace internals {
/// @brief recognizes native symmetric owners and views with policy-driven writes
template <typename T> struct is_native_symmetric : std::false_type { };
/// @brief recognizes every cache policy of a native symmetric owner
template <typename S, int Order, typename P, int O>
struct is_native_symmetric<SymmetricMatrix<S, Order, P, O>> : std::true_type { };
/// @brief recognizes native packed symmetric views
template <typename S, int Order, typename P, int O>
struct is_native_symmetric<SymmetricMatrixView<S, Order, P, O>> : std::true_type { };
/// @brief permits spectral reuse through the owner's controlled cache access
template <typename S, int Order, typename P, int O>
struct is_spectral_cache_source<SymmetricMatrix<S, Order, P, O>> :
    std::bool_constant<(P::Flags & Cache::Spectral::Flags) != 0> { };
/// @brief permits spectral reuse through a view sharing the owner's invalidation slot
template <typename S, int Order, typename P, int O>
struct is_spectral_cache_source<SymmetricMatrixView<S, Order, P, O>> :
    std::bool_constant<(P::Flags & Cache::Spectral::Flags) != 0> { };
}   // namespace internals

template <typename T>
concept NativeSymmetricLike = internals::is_native_symmetric<std::remove_cvref_t<T>>::value;

// detection trait
/// @brief identifies symmetric matrix expressions after removing cv and reference qualifiers
template <typename XprType> struct is_symmetric_matrix {
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<SymmetricMatrixExpr<Type>, Type>;
};
template <typename XprType> static constexpr bool is_symmetric_matrix_v = is_symmetric_matrix<XprType>::value;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SYMMETRIC_H__
