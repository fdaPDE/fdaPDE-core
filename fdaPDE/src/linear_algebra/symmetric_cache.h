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

#ifndef __FDAPDE_LINALG_SYMMETRIC_CACHE_H__
#define __FDAPDE_LINALG_SYMMETRIC_CACHE_H__

#include "header_check.h"

namespace fdapde {

/// @brief owns mutable symmetric coefficients with an optional lazily refreshed spectral cache
template <typename Scalar, int Rows, int Cols, typename Policy = Cache::Spectral> class CachedSymmetricMatrix;
/// @brief borrows symmetric coefficients while routing every write through cache invalidation
template <typename Scalar, int Rows, int Cols, typename Policy = Cache::Spectral> class CachedSymmetricMatrixView;

namespace internals {
/// @brief stores selected cache coefficients and rebuilds their binding on copy
template <typename Slot> struct matrix_cache_owner {
    std::vector<typename Slot::Scalar> values;
    Slot slot;
    /// @brief allocates the exact scalar buffer requested by the slot policy
    explicit matrix_cache_owner(int n) : values(Slot::scalar_count(n)), slot(values.data(), n) { }
    /// @brief copies cached quantities and diagnostics into independently bound storage
    matrix_cache_owner(const matrix_cache_owner& other) : matrix_cache_owner(other.slot.rows()) {
        slot.copy_from(other.slot);
    }
    /// @brief prevents assignment from copying a slot pointer bound to another buffer
    matrix_cache_owner& operator=(const matrix_cache_owner&) = delete;
};

/// @brief borrows a mutable spectral buffer and tracks its validity for symmetric coefficients
template <typename Scalar_, int Order, typename Policy> class symmetric_cache_slot {
   public:
    using Scalar = Scalar_;
    static_assert((Policy::Flags & ~Cache::Spectral::Flags) == 0, "symmetric cache supports None and Spectral only");
    /// @brief binds unused storage without computing any eigenpairs
    symmetric_cache_slot(Scalar* data, int n) : data_(data), n_(n) { }
    /// @brief returns the matrix order associated with the spectral buffer
    int rows() const { return n_; }
    /// @brief counts the selected eigenvector and eigenvalue coefficients
    static std::size_t scalar_count(int n) { return Policy::Flags ? std::size_t(n) * (n + 1) : 0; }
    /// @brief marks eigenpairs stale after a coefficient mutation
    void invalidate() { valid_ = false; }
    /// @brief reports whether the current coefficients have already been decomposed
    bool valid() const { return valid_; }
    /// @brief copies ready or invalid cache state without changing the binding
    void copy_from(const symmetric_cache_slot& other) {
        fdapde_assert(n_ == other.n_, std::invalid_argument, "symmetric cache: incompatible order");
        if (other.valid_) std::copy_n(other.data_, scalar_count(n_), data_);
        valid_ = other.valid_;
    }
    /// @brief prepares eigenpairs once after the latest mutation without assuming positive definiteness
    template <typename Xpr> void prepare(const Xpr& value) {
        if (valid_) return;
        const EVD evd(value);
        const auto vectors = evd.eigenvectors();
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) data_[std::size_t(i) * n_ + j] = vectors(i, j);
            fdapde_strong_assert(
              std::isfinite(evd.eigenvalues()[i]), std::domain_error, "symmetric cache: unrepresentable eigenvalue");
            data_[std::size_t(n_) * n_ + i] = evd.eigenvalues()[i];
        }
        valid_ = true;
    }
    /// @brief borrows read-only eigenvectors after cache preparation
    auto eigenvectors() const {
        fdapde_strong_assert(valid_, std::logic_error, "symmetric cache is stale");
        return MatrixView<const Scalar, Order, Order>(data_, n_, n_);
    }
    /// @brief borrows read-only eigenvalues after cache preparation
    auto eigenvalues() const {
        fdapde_strong_assert(valid_, std::logic_error, "symmetric cache is stale");
        return VectorView<const Scalar, Order>(data_ + std::size_t(n_) * n_, n_);
    }
   private:
    Scalar* data_;
    int n_;
    bool valid_ = false;
};

/// @brief identifies symmetric owners and views with controlled cache-aware writes
template <typename T> struct is_cached_symmetric : std::false_type { };
/// @brief recognizes the owning symmetric cache wrapper
template <typename S, int R, int C, typename P>
struct is_cached_symmetric<CachedSymmetricMatrix<S, R, C, P>> : std::true_type { };
/// @brief recognizes a view that preserves its owner's invalidation mechanism
template <typename S, int R, int C, typename P>
struct is_cached_symmetric<CachedSymmetricMatrixView<S, R, C, P>> : std::true_type { };
/// @brief permits reuse only through an owner that refreshes its own invalidated spectral state
template <typename S, int R, int C, typename P>
struct is_spectral_cache_source<CachedSymmetricMatrix<S, R, C, P>> :
    std::bool_constant<(P::Flags & Cache::Spectral::Flags) != 0> { };
/// @brief permits reuse through a view sharing the owner's cache invalidation slot
template <typename S, int R, int C, typename P>
struct is_spectral_cache_source<CachedSymmetricMatrixView<S, R, C, P>> :
    std::bool_constant<(P::Flags & Cache::Spectral::Flags) != 0> { };
}   // namespace internals

template <typename T>
concept CachedSymmetricLike = internals::is_cached_symmetric<std::remove_cvref_t<T>>::value;

/// @brief owns symmetric packed storage with controlled writes and lazy per-matrix eigenpairs
/// @details raw mutable aliases are unavailable; views and saved coefficient proxies invalidate the shared slot
template <typename S, int R, int C, typename P>
class CachedSymmetricMatrix : public SymmetricMatrixExpr<CachedSymmetricMatrix<S, R, C, P>> {
   public:
    static_assert(
      std::is_floating_point_v<S> && std::same_as<S, std::remove_cv_t<S>>,
      "cached symmetric scalar must be unqualified floating point");
    static_assert((R == Dynamic && C == Dynamic) || (R > 0 && R == C), "cached symmetric shape must be square");
    using Scalar = S;
    using CachePolicy = P;
    using CacheSlot = internals::symmetric_cache_slot<S, R, P>;
    using View = CachedSymmetricMatrixView<S, R, C, P>;
    using ConstView = CachedSymmetricMatrixView<const S, R, C, P>;
    static constexpr int Rows = R, Cols = C, StorageOrder = RowMajor, NestAsRef = 1, ReadOnly = 1;
    using assignment_executor = internals::deleted_assignment_executor;
    /// @brief constructs a fixed zero symmetric matrix with an initially invalid cache
    CachedSymmetricMatrix()
        requires(R != Dynamic)
        : CachedSymmetricMatrix(SymmetricMatrix<S, R, C>()) { }
    /// @brief copies a finite symmetric expression into independent packed coefficients
    template <typename Xpr>
    explicit CachedSymmetricMatrix(const MatrixExpr<Xpr>& source) : data_(checked_storage_(source)) {
        if constexpr (P::Flags) cache_ = std::make_unique<CacheOwner>(rows());
    }
    /// @brief creates a positive-order dynamic zero matrix
    CachedSymmetricMatrix(int rows, int cols)
        requires(R == Dynamic)
        : CachedSymmetricMatrix(ZeroMatrix<S, R, C>(rows, cols)) { }
    /// @brief copies coefficients and any ready spectral quantities with independent ownership
    CachedSymmetricMatrix(const CachedSymmetricMatrix& other) : data_(other.data_) {
        if constexpr (P::Flags) cache_ = std::make_unique<CacheOwner>(*other.cache_);
    }
    /// @brief copies expiring coefficients while preserving valid source views
    CachedSymmetricMatrix(CachedSymmetricMatrix&& other) :
        CachedSymmetricMatrix(static_cast<const CachedSymmetricMatrix&>(other)) { }
    /// @brief replaces an owner through a complete independent candidate
    CachedSymmetricMatrix& operator=(const CachedSymmetricMatrix& other) & {
        if (this != &other) {
            CachedSymmetricMatrix candidate(other);
            commit_(candidate);
        }
        return *this;
    }
    /// @brief rejects replacement of a temporary owner
    void operator=(const CachedSymmetricMatrix&) && = delete;
    /// @brief validates replacement coefficients before changing the owner or its cache
    template <typename Xpr> CachedSymmetricMatrix& operator=(const MatrixExpr<Xpr>& source) & {
        CachedSymmetricMatrix candidate(source);
        commit_(candidate);
        return *this;
    }
    /// @brief returns the current matrix order
    int rows() const { return data_.rows(); }
    /// @brief returns the current matrix order
    int cols() const { return rows(); }
    /// @brief reads a symmetric coefficient without exposing a mutable reference
    S operator()(int i, int j) const { return data_(i, j); }
    /// @brief returns a mutation proxy whose writes invalidate the selected cache
    auto operator()(int i, int j) { return view()(i, j); }
    /// @brief exposes read-only packed coefficients
    const S* data() const { return data_.data(); }
    /// @brief borrows the complete symmetric coefficients without permitting mutation
    auto rep() const& { return SymmetricMatrixView<const S, R, C>(data(), rows(), cols()); }
    /// @brief rejects a coefficient view escaping a temporary owner
    void rep() const&& = delete;
    /// @brief binds mutable coefficient access and cache invalidation to this owner
    View view() & { return View(*this); }
    /// @brief binds read-only coefficients and cache to this owner
    ConstView view() const& { return ConstView(*this); }
    /// @brief rejects a view escaping a temporary owner
    void view() const&& = delete;
    /// @brief refreshes stale spectral quantities before returning their read-only slot
    const CacheSlot& cache() const&
        requires(P::Flags != 0)
    {
        cache_->slot.prepare(rep());
        return cache_->slot;
    }
    /// @brief rejects a cache reference escaping a temporary owner
    void cache() const&&
        requires(P::Flags != 0)
    = delete;
   private:
    template <typename, int, int, typename> friend class CachedSymmetricMatrixView;
    using CacheOwner = internals::matrix_cache_owner<CacheSlot>;
    /// @brief materializes and checks finite square symmetric input before packing it
    template <typename Xpr> static auto checked_storage_(const MatrixExpr<Xpr>& source) {
        fdapde_strong_assert(
          source.rows() > 0 && source.rows() == source.cols() && (R == Dynamic || source.rows() == R),
          std::invalid_argument, "cached symmetric input has incompatible shape");
        const Matrix<S, R, C> dense(source);
        internals::validate_finite_symmetric(dense);
        const S tolerance = S(32) * dense.rows() * std::numeric_limits<S>::epsilon() * dense.inf_norm();
        for (int i = 0; i < dense.rows(); ++i)
            for (int j = 0; j < i; ++j)
                fdapde_strong_assert(
                  std::abs(dense(i, j) - dense(j, i)) <= tolerance, std::invalid_argument,
                  "cached symmetric input must be symmetric");
        return SymmetricMatrix<S, R, C>(dense.template as_symmetric<Lower>());
    }
    /// @brief copies candidate coefficients before replacing the associated cache ownership
    void commit_(CachedSymmetricMatrix& candidate) {
        data_ = candidate.data_;
        if constexpr (P::Flags) {
            if (cache_->slot.rows() == candidate.rows())
                cache_->slot.copy_from(candidate.cache_->slot);
            else
                cache_.swap(candidate.cache_);
        }
    }
    SymmetricMatrix<S, R, C> data_;
    [[no_unique_address]] std::conditional_t<P::Flags == 0, internals::empty_spd_cache<3>, std::unique_ptr<CacheOwner>>
      cache_;
};

/// @brief borrows packed symmetric coefficients and routes writes through their shared cache slot
/// @details owner shape changes or destruction invalidate the view and its saved coefficient proxies
template <typename S, int R, int C, typename P>
class CachedSymmetricMatrixView : public SymmetricMatrixExpr<CachedSymmetricMatrixView<S, R, C, P>> {
   public:
    using Scalar = std::remove_const_t<S>;
    using Owner = CachedSymmetricMatrix<Scalar, R, C, P>;
    using CachePolicy = P;
    using CacheSlot = typename Owner::CacheSlot;
    using View = CachedSymmetricMatrixView<Scalar, R, C, P>;
    using ConstView = CachedSymmetricMatrixView<const Scalar, R, C, P>;
    static constexpr int Rows = R, Cols = C, StorageOrder = RowMajor, NestAsRef = 0, ReadOnly = 1;
    using assignment_executor = internals::deleted_assignment_executor;
    /// @brief retains the external coefficient and invalidation bindings
    CachedSymmetricMatrixView(const CachedSymmetricMatrixView&) = default;
    /// @brief borrows coefficients and their cache from a persistent mutable owner
    explicit CachedSymmetricMatrixView(Owner& owner) : data_(owner.data_.data()), n_(owner.rows()) {
        if constexpr (P::Flags) cache_ = &owner.cache_->slot;
    }
    /// @brief borrows read-only coefficients and a lazily prepared cache from a persistent owner
    explicit CachedSymmetricMatrixView(const Owner& owner)
        requires(std::is_const_v<S>)
        : data_(owner.data()), n_(owner.rows()) {
        if constexpr (P::Flags) cache_ = &owner.cache_->slot;
    }
    /// @brief rejects a binding into an expiring owner
    CachedSymmetricMatrixView(Owner&&) = delete;
    /// @brief rejects a binding into an expiring const owner
    CachedSymmetricMatrixView(const Owner&&) = delete;
    /// @brief converts mutable value access to a read-only binding without computing cached quantities
    CachedSymmetricMatrixView(const View& other)
        requires(std::is_const_v<S>)
        : data_(other.data_), n_(other.n_), cache_(other.cache_) { }
    /// @brief returns the bound matrix order
    int rows() const { return n_; }
    /// @brief returns the bound matrix order
    int cols() const { return n_; }
    /// @brief reads a packed coefficient through its symmetric interpretation
    Scalar operator()(int i, int j) const { return rep()(i, j); }
    /// @brief exposes read-only packed coefficients
    const Scalar* data() const { return data_; }
    /// @brief borrows a read-only ordinary symmetric view
    auto rep() const { return SymmetricMatrixView<const Scalar, R, C>(data_, n_, n_); }
    /// @brief prepares eigenpairs against the current coefficients before borrowing them
    const CacheSlot& cache() const
        requires(P::Flags != 0)
    {
        cache_->prepare(rep());
        return *cache_;
    }
    /// @brief tracks one coefficient alias so each later write invalidates the current cache
    class coefficient_proxy {
       public:
        /// @brief retains the scalar address and its optional invalidation slot
        coefficient_proxy(Scalar* value, CacheSlot* cache) : value_(value), cache_(cache) { }
        /// @brief reads the current coefficient
        operator Scalar() const { return *value_; }
        /// @brief stores a finite coefficient and invalidates any ready eigenpairs
        coefficient_proxy& operator=(Scalar value) {
            fdapde_strong_assert(std::isfinite(value), std::invalid_argument, "symmetric coefficients must be finite");
            *value_ = value;
            if constexpr (P::Flags) cache_->invalidate();
            return *this;
        }
        /// @brief copies the referenced value rather than rebinding a saved coefficient alias
        coefficient_proxy& operator=(const coefficient_proxy& other) { return *this = Scalar(other); }
       private:
        Scalar* value_;
        CacheSlot* cache_;
    };
    /// @brief binds a symmetric coefficient to cache-aware mutable access
    auto operator()(int i, int j)
        requires(!std::is_const_v<S>)
    {
        internals::validate_matrix_index(i, j, n_, n_);
        if (i < j) std::swap(i, j);
        return coefficient_proxy(data_ + std::size_t(i) * (i + 1) / 2 + j, cache_);
    }
    /// @brief validates replacement coefficients before copying into this fixed-shape view
    template <typename Xpr>
    CachedSymmetricMatrixView& assign(const MatrixExpr<Xpr>& source)
        requires(!std::is_const_v<S>)
    {
        fdapde_strong_assert(
          source.rows() == n_ && source.cols() == n_, std::invalid_argument,
          "symmetric view assignment cannot change shape");
        const Owner candidate(source);
        std::copy_n(candidate.data(), std::size_t(n_) * (n_ + 1) / 2, data_);
        if constexpr (P::Flags) cache_->invalidate();
        return *this;
    }
    /// @brief assigns a complete expression while preserving the view binding
    template <typename Xpr>
    CachedSymmetricMatrixView& operator=(const MatrixExpr<Xpr>& source)
        requires(!std::is_const_v<S>)
    {
        return assign(source);
    }
    /// @brief copies the source value while retaining this view's binding
    CachedSymmetricMatrixView& operator=(const CachedSymmetricMatrixView& source)
        requires(!std::is_const_v<S>)
    {
        return assign(source);
    }
    /// @brief forbids value replacement through a const view
    void operator=(const CachedSymmetricMatrixView&)
        requires(std::is_const_v<S>)
    = delete;
   private:
    template <typename, int> friend class MatrixBatch;
    template <typename, int, int, typename> friend class CachedSymmetricMatrixView;
    /// @brief binds a batch-owned packed coefficient row and its invalidation slot
    CachedSymmetricMatrixView(S* data, int n, CacheSlot* cache) : data_(data), n_(n), cache_(cache) { }
    S* data_;
    int n_;
    CacheSlot* cache_ = nullptr;
};
}   // namespace fdapde
#endif
