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

#ifndef __FDAPDE_LINALG_ROTATION_H__
#define __FDAPDE_LINALG_ROTATION_H__
#include "header_check.h"
#include "rotation_schur.h"

namespace fdapde {
namespace RotationCache {
/// @brief selects reusable real rotation planes and the unique minimum logarithm independently
template <unsigned Bits> struct Policy {
    static_assert((Bits & ~3u) == 0, "rotation cache supports Schur and Log only");
    static constexpr unsigned Flags = Bits;
};
using None = Policy<0>;
using Schur = Policy<1>;
using Log = Policy<2>;
template <typename... Policies> using Union = Policy<(Policies::Flags | ... | 0u)>;
}   // namespace RotationCache
/// @brief owns a verified orientation-preserving orthogonal matrix and its selected algebraic cache
template <typename S, int R, int C, typename P = RotationCache::None> class RotationMatrix;
/// @brief borrows a verified rotation with joint coefficient and cache replacement
template <typename S, int R, int C, typename P = RotationCache::None> class RotationMatrixView;
namespace internals {
/// @brief recognizes only verified rotation owners and their controlled views
template <typename T> struct is_rotation : std::false_type { };
/// @brief recognizes the checked owning rotation type
template <typename S, int R, int C, typename P> struct is_rotation<RotationMatrix<S, R, C, P>> : std::true_type { };
/// @brief recognizes views created exclusively from checked rotation storage
template <typename S, int R, int C, typename P> struct is_rotation<RotationMatrixView<S, R, C, P>> : std::true_type { };

/// @brief binds selected real planes and compact logarithms to an external cache buffer
template <typename S, int N, typename P> class rotation_cache_slot {
   public:
    using Scalar = S;
    static_assert((P::Flags & ~3u) == 0, "rotation cache supports Schur and Log only");
    static constexpr bool HasSchur = (P::Flags & 1) != 0, HasLog = (P::Flags & 2) != 0;
    /// @brief binds the selected cache storage to a matrix order
    rotation_cache_slot(S* data, int n) : data_(data), n_(n) { }
    /// @brief returns the matrix order
    int rows() const { return n_; }
    /// @brief counts only the selected algebraic quantities without retaining unrequested Schur factors
    static std::size_t scalar_count(int n) {
        return (HasSchur ? std::size_t(n) * (n + 1) : 0) +
               (HasLog ? std::max(std::size_t(1), std::size_t(n) * (n - 1) / 2) : 0);
    }
    /// @brief copies the full cache and diagnostics while preserving this slot binding
    void copy_from(const rotation_cache_slot& source) {
        fdapde_assert(n_ == source.n_, std::invalid_argument, "rotation cache: incompatible order");
        std::copy_n(source.data_, scalar_count(n_), data_);
        diagnostics_ = source.diagnostics_;
        distance_ = source.distance_;
        planes_available_ = source.planes_available_;
    }
    /// @brief keeps valid rotations constructible when a numerical plane decomposition is unresolved
    template <typename Q> void prepare(const Q& q) {
        try {
            prepare_planes_(q);
        } catch (const std::domain_error&) { unresolved_(q); } catch (const std::runtime_error&) {
            unresolved_(q);
        }
    }
   private:
    /// @brief records numerical failure without turning it into a failed rotation invariant check
    template <typename Q> void unresolved_(const Q& q) {
        planes_available_ = false;
        diagnostics_ = {
          RotationLogStatus::Unresolved, std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity()};
        distance_ = std::numeric_limits<double>::quiet_NaN();
        try {
            distance_ = rotation_distance_cosines(q);
        } catch (const std::domain_error&) {
        } catch (const std::runtime_error&) { }
    }
    /// @brief prepares all requested quantities from one real plane decomposition
    template <typename Q> void prepare_planes_(const Q& q) {
        const rotation_schur<S, N> schur(q);
        planes_available_ = true;
        diagnostics_ = schur.diagnostics();
        distance_ = 0;
        for (int k = 0; k < n_; ++k) distance_ = std::hypot(distance_, double(schur.angles()[k]));
        if constexpr (HasSchur) {
            for (int i = 0; i < n_; ++i) {
                for (int j = 0; j < n_; ++j) data_[std::size_t(i) * n_ + j] = schur.vectors()(i, j);
                data_[std::size_t(n_) * n_ + i] = schur.angles()[i];
            }
        }
        if constexpr (HasLog) {
            std::fill_n(data_ + log_offset_(), std::size_t(n_) * (n_ - 1) / 2, S(0));
            if (diagnostics_.unique()) {
                const auto logarithm = rotation_schur<S, N>::logarithm(schur.vectors(), schur.angles());
                std::copy_n(logarithm.data(), std::size_t(n_) * (n_ - 1) / 2, data_ + log_offset_());
            }
        }
    }
   public:
    /// @brief initializes known identity planes and a zero logarithm without a decomposition
    void set_identity() {
        std::fill_n(data_, scalar_count(n_), S(0));
        if constexpr (HasSchur)
            for (int i = 0; i < n_; ++i) data_[std::size_t(i) * n_ + i] = S(1);
        planes_available_ = true;
        diagnostics_ = {};
        distance_ = 0;
    }
    /// @brief borrows the real orthogonal basis when Schur storage was selected
    auto vectors() const
        requires(HasSchur)
    {
        fdapde_strong_assert(planes_available_, std::domain_error, "rotation Schur factors are numerically unresolved");
        return MatrixView<const S, N, N>(data_, n_, n_);
    }
    /// @brief borrows adjacent signed plane angles when Schur storage was selected
    auto angles() const
        requires(HasSchur)
    {
        fdapde_strong_assert(planes_available_, std::domain_error, "rotation Schur angles are numerically unresolved");
        return VectorView<const S, N>(data_ + std::size_t(n_) * n_, n_);
    }
    /// @brief borrows a compact logarithm only when its branch is uniquely resolved
    auto logarithm() const
        requires(HasLog)
    {
        fdapde_strong_assert(diagnostics_.unique(), std::domain_error, "rotation logarithm is ambiguous or unresolved");
        return SkewSymmetricMatrixView<const S, N, N>(data_ + log_offset_(), n_, n_);
    }
    /// @brief returns branch diagnostics even when no ordinary logarithm is available
    RotationLogDiagnostics diagnostics() const { return diagnostics_; }
    /// @brief returns the full-Frobenius distance to identity even at the cut locus
    double distance() const {
        fdapde_strong_assert(
          std::isfinite(distance_), std::domain_error, "rotation distance is numerically unresolved");
        return distance_;
    }
   private:
    /// @brief locates the packed logarithm after optional real Schur coefficients
    std::size_t log_offset_() const { return HasSchur ? std::size_t(n_) * (n_ + 1) : 0; }
    S* data_;
    int n_;
    RotationLogDiagnostics diagnostics_;
    double distance_ = 0;
    bool planes_available_ = false;
};
}   // namespace internals
template <typename T>
concept RotationLike = internals::is_rotation<std::remove_cvref_t<T>>::value;

/// @brief owns finite rotations checked for orthogonality and positive determinant
/// @details coefficient access is read-only and every replacement commits coefficients and cache together
template <typename S, int R, int C, typename P>
class RotationMatrix : public OrthogonalMatrixExpr<RotationMatrix<S, R, C, P>> {
   public:
    static_assert(
      std::is_floating_point_v<S> && std::same_as<S, std::remove_cv_t<S>>,
      "rotation scalar must be unqualified floating point");
    static_assert((R == Dynamic && C == Dynamic) || (R > 0 && R == C), "rotation shape must be square");
    using Scalar = S;
    using CachePolicy = P;
    using CacheSlot = internals::rotation_cache_slot<S, R, P>;
    using View = RotationMatrixView<S, R, C, P>;
    using ConstView = RotationMatrixView<const S, R, C, P>;
    static constexpr int Rows = R, Cols = C, StorageOrder = RowMajor, NestAsRef = 1, ReadOnly = 1;
    using assignment_executor = internals::deleted_assignment_executor;
    /// @brief rejects an owner with no verified coefficients
    RotationMatrix() = delete;
    /// @brief materializes and validates input before preparing the selected cache
    template <typename Xpr> explicit RotationMatrix(const MatrixExpr<Xpr>& source) : data_(checked_storage_(source)) {
        if constexpr (P::Flags) {
            cache_ = std::make_unique<CacheOwner>(rows());
            cache_->slot.prepare(data_);
        }
    }
    /// @brief copies coefficients and ready caches with independent ownership
    RotationMatrix(const RotationMatrix& other) : data_(other.data_) {
        if constexpr (P::Flags) cache_ = std::make_unique<CacheOwner>(*other.cache_);
    }
    /// @brief copies expiring storage while preserving valid source coefficients
    RotationMatrix(RotationMatrix&& other) : RotationMatrix(static_cast<const RotationMatrix&>(other)) { }
    /// @brief prepares an independent replacement before committing the complete value
    RotationMatrix& operator=(const RotationMatrix& other) & {
        if (this != &other) {
            RotationMatrix candidate(other);
            commit_(candidate);
        }
        return *this;
    }
    /// @brief rejects replacement of a temporary owner
    void operator=(const RotationMatrix&) && = delete;
    /// @brief validates a candidate and its cache before replacing the owner
    template <typename Xpr> RotationMatrix& operator=(const MatrixExpr<Xpr>& source) & {
        RotationMatrix candidate(source);
        commit_(candidate);
        return *this;
    }
    /// @brief creates a verified identity and its known cache without Schur computation
    static RotationMatrix Identity(int n = R) {
        fdapde_strong_assert(
          n > 0 && (R == Dynamic || n == R), std::invalid_argument, "rotation identity: invalid order");
        return RotationMatrix(n, identity_t {});
    }
    /// @brief returns the matrix order
    int rows() const { return data_.rows(); }
    /// @brief returns the matrix order
    int cols() const { return data_.cols(); }
    /// @brief reads a coefficient without permitting invariant-breaking writes
    S operator()(int i, int j) const { return data_(i, j); }
    /// @brief borrows read-only row-major coefficients
    const S* data() const { return data_.data(); }
    /// @brief returns a checked inverse by transposing the orthogonal coefficients
    RotationMatrix inv() const { return RotationMatrix(data_.transpose()); }
    /// @brief borrows a persistent owner for joint value and cache updates
    View view() & { return View(*this); }
    /// @brief borrows read-only coefficients and cache from a persistent owner
    ConstView view() const& { return ConstView(*this); }
    /// @brief rejects a view escaping a temporary owner
    void view() const&& = delete;
    /// @brief borrows the prepared cache while the current owner value remains alive
    const CacheSlot& cache() const&
        requires(P::Flags != 0)
    {
        return cache_->slot;
    }
    /// @brief rejects a cache reference escaping a temporary owner
    void cache() const&&
        requires(P::Flags != 0)
    = delete;
   private:
    template <typename, int, int, typename> friend class RotationMatrixView;
    using CacheOwner = internals::matrix_cache_owner<CacheSlot>;
    /// @brief restricts the known-identity construction path to the owning factory
    struct identity_t { };
    /// @brief initializes exact identity coefficients and selected cached quantities
    RotationMatrix(int n, identity_t) : data_(IdentityMatrix<S, R, C>(n, n)) {
        if constexpr (P::Flags) {
            cache_ = std::make_unique<CacheOwner>(n);
            cache_->slot.set_identity();
        }
    }
    /// @brief validates dimensions and the SO(n) invariants before publishing coefficients
    template <typename Xpr> static auto checked_storage_(const MatrixExpr<Xpr>& source) {
        fdapde_strong_assert(
          source.rows() > 0 && source.rows() == source.cols() && (R == Dynamic || source.rows() == R),
          std::invalid_argument, "rotation: invalid square shape");
        const Matrix<S, R, C> value(source);
        fdapde_strong_assert(
          internals::is_orthogonal(value), std::domain_error, "rotation: coefficients must be finite and orthogonal");
        fdapde_strong_assert(value.determinant() > S(0), std::domain_error, "rotation: reflections are not in SO(n)");
        return value;
    }
    /// @brief replaces coefficients before transferring their associated cache ownership
    void commit_(RotationMatrix& candidate) {
        data_ = candidate.data_;
        if constexpr (P::Flags) {
            if (cache_->slot.rows() == candidate.rows())
                cache_->slot.copy_from(candidate.cache_->slot);
            else
                cache_.swap(candidate.cache_);
        }
    }
    Matrix<S, R, C> data_;
    [[no_unique_address]] std::conditional_t<P::Flags == 0, internals::empty_spd_cache<4>, std::unique_ptr<CacheOwner>>
      cache_;
};

/// @brief borrows verified dense coefficients and preserves the SO(n) contract on assignment
/// @details owner destruction or shape changes invalidate views; view assignment keeps its original binding
template <typename S, int R, int C, typename P>
class RotationMatrixView : public OrthogonalMatrixExpr<RotationMatrixView<S, R, C, P>> {
   public:
    using Scalar = std::remove_const_t<S>;
    using Owner = RotationMatrix<Scalar, R, C, P>;
    using CachePolicy = P;
    using CacheSlot = typename Owner::CacheSlot;
    using View = RotationMatrixView<Scalar, R, C, P>;
    using ConstView = RotationMatrixView<const Scalar, R, C, P>;
    static constexpr int Rows = R, Cols = C, StorageOrder = RowMajor, NestAsRef = 0, ReadOnly = 1;
    using assignment_executor = internals::deleted_assignment_executor;
    /// @brief copies a verified view binding
    RotationMatrixView(const RotationMatrixView&) = default;
    /// @brief borrows a persistent mutable owner for whole-value replacement
    explicit RotationMatrixView(Owner& owner) : data_(owner.data_.data()), n_(owner.rows()) {
        if constexpr (P::Flags) cache_ = &owner.cache_->slot;
    }
    /// @brief borrows a persistent const owner's coefficients and cache
    explicit RotationMatrixView(const Owner& owner)
        requires(std::is_const_v<S>)
        : data_(owner.data()), n_(owner.rows()) {
        if constexpr (P::Flags) cache_ = &owner.cache_->slot;
    }
    /// @brief rejects a view into an expiring owner
    RotationMatrixView(Owner&&) = delete;
    /// @brief rejects a view into an expiring const owner
    RotationMatrixView(const Owner&&) = delete;
    /// @brief converts mutable value access to a read-only binding without computing cached quantities
    RotationMatrixView(const View& other)
        requires(std::is_const_v<S>)
        : data_(other.data_), n_(other.n_), cache_(other.cache_) { }
    /// @brief returns the bound matrix order
    int rows() const { return n_; }
    /// @brief returns the bound matrix order
    int cols() const { return n_; }
    /// @brief reads the verified dense coefficient after checking its indices
    Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, n_, n_);
        return data_[std::size_t(i) * n_ + j];
    }
    /// @brief borrows read-only dense coefficients
    const Scalar* data() const { return data_; }
    /// @brief returns the checked transpose as an owning inverse
    Owner inv() const { return Owner(this->transpose()); }
    /// @brief borrows the current prepared cache without extending its lifetime
    const CacheSlot& cache() const
        requires(P::Flags != 0)
    {
        return *cache_;
    }
    /// @brief validates a whole replacement before copying coefficients and selected quantities into the binding
    template <typename Xpr>
    RotationMatrixView& assign(const MatrixExpr<Xpr>& source)
        requires(!std::is_const_v<S>)
    {
        fdapde_strong_assert(
          source.rows() == n_ && source.cols() == n_, std::invalid_argument, "rotation view: incompatible shape");
        const Owner candidate(source);
        std::copy_n(candidate.data(), std::size_t(n_) * n_, data_);
        if constexpr (P::Flags) cache_->copy_from(candidate.cache());
        return *this;
    }
    /// @brief assigns an expression without rebinding this view
    template <typename Xpr>
    RotationMatrixView& operator=(const MatrixExpr<Xpr>& source)
        requires(!std::is_const_v<S>)
    {
        return assign(source);
    }
    /// @brief copies a source view's complete value while retaining this binding
    RotationMatrixView& operator=(const RotationMatrixView& source)
        requires(!std::is_const_v<S>)
    {
        return assign(source);
    }
    /// @brief forbids replacement through a read-only view
    void operator=(const RotationMatrixView&)
        requires(std::is_const_v<S>)
    = delete;
   private:
    template <typename, int> friend class MatrixBatch;
    template <typename, int, int, typename> friend class RotationMatrixView;
    /// @brief binds batch-owned verified coefficients and their optional cache slot
    RotationMatrixView(S* data, int n, CacheSlot* cache) : data_(data), n_(n), cache_(cache) { }
    S* data_;
    int n_;
    CacheSlot* cache_ = nullptr;
};

/// @brief composes two verified rotations and checks the rounded product before publishing it
template <RotationLike L, RotationLike R> auto operator*(const L& left, const R& right) {
    using S = promote_type_t<typename L::Scalar, typename R::Scalar>;
    const Matrix<S, L::Rows, L::Cols> a(left);
    const Matrix<S, R::Rows, R::Cols> b(right);
    return RotationMatrix<S, L::Rows, R::Cols, typename L::CachePolicy>(a * b);
}
/// @brief returns a minimum skew logarithm together with branch diagnostics
template <typename S, int N> struct RotationLogResult {
    SkewSymmetricMatrix<S, N, N> tangent;
    RotationLogDiagnostics diagnostics;
};
/// @brief explicitly selects a minimum logarithm, including a deterministic branch at the cut locus
template <RotationLike Q> auto minimum_rotation_log(const Q& q) {
    using S = typename Q::Scalar;
    constexpr int N = Q::Rows;
    if constexpr ((Q::CachePolicy::Flags & 2) != 0) {
        if (q.cache().diagnostics().unique())
            return RotationLogResult<S, N> {
              SkewSymmetricMatrix<S, N, N>(q.cache().logarithm()), q.cache().diagnostics()};
    }
    if constexpr ((Q::CachePolicy::Flags & 1) != 0)
        return RotationLogResult<S, N> {
          internals::rotation_schur<S, N>::logarithm(q.cache().vectors(), q.cache().angles()), q.cache().diagnostics()};
    else {
        const internals::rotation_schur<S, N> schur(q);
        return RotationLogResult<S, N> {
          internals::rotation_schur<S, N>::logarithm(schur.vectors(), schur.angles()), schur.diagnostics()};
    }
}
/// @brief returns the unique minimum logarithm or rejects an ambiguous or unresolved branch
template <RotationLike Q> auto rotation_log(const Q& q) {
    using S = typename Q::Scalar;
    constexpr int N = Q::Rows;
    if constexpr ((Q::CachePolicy::Flags & 2) != 0)
        return SkewSymmetricMatrix<S, N, N>(q.cache().logarithm());
    else if constexpr ((Q::CachePolicy::Flags & 1) != 0) {
        fdapde_strong_assert(
          q.cache().diagnostics().unique(), std::domain_error, "rotation logarithm is ambiguous or unresolved");
        return internals::rotation_schur<S, N>::logarithm(q.cache().vectors(), q.cache().angles());
    } else {
        const auto result = minimum_rotation_log(q);
        fdapde_strong_assert(
          result.diagnostics.unique(), std::domain_error, "rotation logarithm is ambiguous or unresolved");
        return result.tangent;
    }
}
/// @brief returns the full-Frobenius rotation distance from identity, including the cut locus
template <RotationLike Q> double rotation_distance_identity(const Q& q) {
    if constexpr (Q::CachePolicy::Flags != 0)
        return q.cache().distance();
    else {
        try {
            const internals::rotation_schur<typename Q::Scalar, Q::Rows> schur(q);
            double norm = 0;
            for (int i = 0; i < q.rows(); ++i) norm = std::hypot(norm, double(schur.angles()[i]));
            return norm;
        } catch (const std::domain_error&) {
            return internals::rotation_distance_cosines(q);
        } catch (const std::runtime_error&) { return internals::rotation_distance_cosines(q); }
    }
}
/// @brief maps a finite skew tangent to a verified rotation with the requested cache policy
template <typename P = RotationCache::None, typename Xpr>
auto rotation_exp(const SkewSymmetricMatrixExpr<Xpr>& tangent, double step = 1) {
    return RotationMatrix<std::remove_cv_t<typename Xpr::Scalar>, Xpr::Rows, Xpr::Cols, P>(
      internals::skew_exponential(tangent, step));
}
}   // namespace fdapde
#endif
