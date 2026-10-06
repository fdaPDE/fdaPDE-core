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

#ifndef __FDAPDE_LINALG_SPD_CACHE_H__
#define __FDAPDE_LINALG_SPD_CACHE_H__

#include <memory>
#include <vector>

#include "header_check.h"

namespace fdapde {
namespace Cache {

/// @brief selects the algebraic quantities retained by an SPD value at compile time
template <unsigned Flags_> struct Policy {
    fdapde_static_assert((Flags_ & ~127u) == 0, SPD_CACHE_POLICY_CONTAINS_UNKNOWN_FLAGS);
    static constexpr unsigned Flags = Flags_;
};
using None = Policy<0>;
using Spectral = Policy<1>;
using Log = Policy<2>;
using Sqrt = Policy<4>;
using InverseSqrt = Policy<8>;
using LogDividedDifferences = Policy<16>;
using Cholesky = Policy<32>;
using LogCholesky = Policy<64>;
template <typename... Policies> using Union = Policy<(Policies::Flags | ... | 0u)>;

}   // namespace Cache

/// @brief identifies reusable per-point work requested by a geometry
enum class Usage : unsigned {
    None = 0,
    Distance = 1,
    InterpolationNodes = 2,
    TangentMetric = 4,
    LogExpDifferentials = 8,
    BasePointMaps = 16
};
/// @brief combines independent geometric uses without duplicating common cache quantities
constexpr Usage operator|(Usage lhs, Usage rhs) {
    return static_cast<Usage>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

namespace internals {

/// @brief evaluates the stable logarithmic divided difference
template <typename Scalar> Scalar log_divided_difference(Scalar x, Scalar y);

/// @brief supplies storage-free state for a disabled cache
template <int Tag = 0> struct empty_spd_cache { };

template <typename Policy, typename Quantity>
inline constexpr bool spd_cache_has_v = (Policy::Flags & Quantity::Flags) == Quantity::Flags;

/// @brief views one cache slot whose selected quantities occupy an external contiguous scalar buffer
/// @details logarithmic divided differences may be applied with retained eigenvectors only when spectral is selected
template <typename Scalar_, int Order_, typename Policy_> class spd_cache_slot {
   public:
    using Scalar = Scalar_;
    using CachePolicy = Policy_;
    static constexpr int Rows = Order_;
    static constexpr int Cols = Order_;
    template <typename Quantity> static constexpr bool Has = spd_cache_has_v<CachePolicy, Quantity>;

    /// @brief binds the slot to its allocated buffer without evaluating any matrix operation
    spd_cache_slot(Scalar* data, int order) : data_(data), order_(order) { }
    /// @brief returns the uniform matrix order of this slot
    int rows() const { return order_; }
    /// @brief returns the number of scalar coefficients retained for the requested order
    static std::size_t scalar_count(int n) {
        const auto order = static_cast<std::size_t>(n);
        return (Has<Cache::Spectral> ? order * (order + 1) : 0) +
               (Has<Cache::Log> + Has<Cache::Sqrt> + Has<Cache::InverseSqrt>)*order * (order + 1) / 2 +
               (Has<Cache::LogDividedDifferences> + Has<Cache::Cholesky>)*order * order +
               (Has<Cache::LogCholesky> ? order * (order + 1) / 2 : 0);
    }
    /// @brief returns read-only eigenvectors in the basis shared by the retained eigenvalues
    auto eigenvectors() const
        requires(Has<Cache::Spectral>)
    {
        return MatrixView<const Scalar, Rows, Cols>(data_, order_, order_);
    }
    /// @brief returns read-only eigenvalues associated with the retained eigenvector columns
    auto eigenvalues() const
        requires(Has<Cache::Spectral>)
    {
        return VectorView<const Scalar, Rows>(data_ + std::size_t(order_) * order_, order_);
    }
    /// @brief borrows the selected symmetric matrix quantity from packed storage
    template <typename Quantity>
    auto matrix() const
        requires(Has<Quantity>)
    {
        static_assert(
          Quantity::Flags == 2 || Quantity::Flags == 4 || Quantity::Flags == 8 || Quantity::Flags == 64,
          "cache matrix access requires log, sqrt, inverse sqrt or log-Cholesky");
        return SymmetricMatrixView<const Scalar, Rows, Cols>(data_ + offset_<Quantity>(), order_, order_);
    }
    /// @brief borrows the logarithmic divided differences in the cache's original spectral ordering
    auto log_divided_differences() const
        requires(Has<Cache::LogDividedDifferences>)
    {
        return MatrixView<const Scalar, Rows, Cols>(data_ + offset_<Cache::LogDividedDifferences>(), order_, order_);
    }
    /// @brief borrows the positive-diagonal lower factor with a zero upper triangle
    auto cholesky() const
        requires(Has<Cache::Cholesky>)
    {
        return MatrixView<const Scalar, Rows, Cols>(data_ + offset_<Cache::Cholesky>(), order_, order_);
    }
    /// @brief exposes the read-only scalar buffer for layout inspection
    const Scalar* data() const { return data_; }

    /// @brief copies a same-policy cache into this slot without changing either binding
    void copy_from(const spd_cache_slot& source) {
        fdapde_assert(order_ == source.order_, std::invalid_argument, "SPD cache: incompatible slot dimensions");
        std::copy_n(source.data_, scalar_count(order_), data_);
    }
    /// @brief copies common quantities while preserving the association between eigenvectors and divided differences
    template <bool CoherentBasis = true, int OtherOrder, typename OtherPolicy>
    void copy_common(const spd_cache_slot<Scalar, OtherOrder, OtherPolicy>& source) {
        if constexpr (Has<Cache::Spectral> && spd_cache_has_v<OtherPolicy, Cache::Spectral>) {
            const auto vectors = source.eigenvectors();
            const auto values = source.eigenvalues();
            for (int i = 0; i < order_; ++i) {
                for (int j = 0; j < order_; ++j) data_[std::size_t(i) * order_ + j] = vectors(i, j);
                data_[std::size_t(order_) * order_ + i] = values[i];
            }
        }
        if constexpr (Has<Cache::Log> && spd_cache_has_v<OtherPolicy, Cache::Log>) copy_matrix_<Cache::Log>(source);
        if constexpr (Has<Cache::Sqrt> && spd_cache_has_v<OtherPolicy, Cache::Sqrt>) copy_matrix_<Cache::Sqrt>(source);
        if constexpr (Has<Cache::LogCholesky> && spd_cache_has_v<OtherPolicy, Cache::LogCholesky>)
            copy_matrix_<Cache::LogCholesky>(source);
        if constexpr (Has<Cache::Cholesky> && spd_cache_has_v<OtherPolicy, Cache::Cholesky>)
            std::copy_n(source.cholesky().data(), std::size_t(order_) * order_, data_ + offset_<Cache::Cholesky>());
        if constexpr (Has<Cache::InverseSqrt> && spd_cache_has_v<OtherPolicy, Cache::InverseSqrt>)
            copy_matrix_<Cache::InverseSqrt>(source);
        if constexpr (
          CoherentBasis && Has<Cache::LogDividedDifferences> &&
          spd_cache_has_v<OtherPolicy, Cache::LogDividedDifferences>) {
            const auto matrix = source.log_divided_differences();
            std::copy_n(matrix.data(), std::size_t(order_) * order_, data_ + offset_<Cache::LogDividedDifferences>());
        }
    }
    /// @brief fills known identity intermediates without an eigendecomposition
    void set_identity() {
        std::fill_n(data_, scalar_count(order_), Scalar(0));
        if constexpr (Has<Cache::Spectral>) {
            for (int i = 0; i < order_; ++i) {
                data_[std::size_t(i) * order_ + i] = Scalar(1);
                data_[std::size_t(order_) * order_ + i] = Scalar(1);
            }
        }
        if constexpr (Has<Cache::Sqrt>) set_identity_matrix_<Cache::Sqrt>();
        if constexpr (Has<Cache::InverseSqrt>) set_identity_matrix_<Cache::InverseSqrt>();
        if constexpr (Has<Cache::Cholesky>)
            for (int i = 0; i < order_; ++i)
                data_[offset_<Cache::Cholesky>() + std::size_t(i) * order_ + i] = Scalar(1);
        if constexpr (Has<Cache::LogDividedDifferences>) {
            std::fill_n(data_ + offset_<Cache::LogDividedDifferences>(), std::size_t(order_) * order_, Scalar(1));
        }
    }
    /// @brief prepares selected quantities absent from the reusable policy using one certified eigendecomposition
    template <typename ReusablePolicy = Cache::None, typename Spectral> void prepare(const Spectral& spectral) {
        using Missing = Cache::Policy<CachePolicy::Flags & ~ReusablePolicy::Flags>;
        const auto vectors = spectral.eigenvectors();
        const auto& values = spectral.eigenvalues();
        if constexpr (spd_cache_has_v<Missing, Cache::Spectral>) {
            for (int i = 0; i < order_; ++i) {
                for (int j = 0; j < order_; ++j) data_[std::size_t(i) * order_ + j] = vectors(i, j);
                data_[std::size_t(order_) * order_ + i] = values[i];
            }
        }
        if constexpr (spd_cache_has_v<Missing, Cache::Log>)
            prepare_matrix_<Cache::Log>(spectral, [](Scalar x) { return std::log(x); });
        if constexpr (spd_cache_has_v<Missing, Cache::Sqrt>)
            prepare_matrix_<Cache::Sqrt>(spectral, [](Scalar x) { return std::sqrt(x); });
        if constexpr (spd_cache_has_v<Missing, Cache::InverseSqrt>)
            prepare_matrix_<Cache::InverseSqrt>(spectral, [](Scalar x) { return Scalar(1) / std::sqrt(x); });
        if constexpr (spd_cache_has_v<Missing, Cache::LogDividedDifferences>) {
            for (int i = 0; i < order_; ++i) {
                for (int j = 0; j < order_; ++j) {
                    const Scalar x = values[i], y = values[j];
                    const Scalar value = log_divided_difference(x, y);
                    fdapde_strong_assert(
                      std::isfinite(value), std::domain_error, "SPD cache: nonfinite divided difference");
                    data_[offset_<Cache::LogDividedDifferences>() + std::size_t(i) * order_ + j] = value;
                }
            }
        }
    }
    /// @brief prepares absent triangular quantities from the owner's certified coefficients
    template <typename ReusablePolicy = Cache::None, typename Coefficients>
    void prepare_triangular(const Coefficients& coefficients) {
        using Missing = Cache::Policy<CachePolicy::Flags & ~ReusablePolicy::Flags>;
        if constexpr (spd_cache_has_v<Missing, Cache::Cholesky> || spd_cache_has_v<Missing, Cache::LogCholesky>) {
            Matrix<Scalar, Rows, Cols> temporary;
            if constexpr (Rows == Dynamic && !Has<Cache::Cholesky>) temporary.resize(order_, order_);
            auto factor = [&]() {
                if constexpr (Has<Cache::Cholesky>)
                    return MatrixView<Scalar, Rows, Cols>(data_ + offset_<Cache::Cholesky>(), order_, order_);
                else
                    return MatrixView<Scalar, Rows, Cols>(temporary.data(), order_, order_);
            }();
            if constexpr (!spd_cache_has_v<ReusablePolicy, Cache::Cholesky> || !Has<Cache::Cholesky>) {
                for (int i = 0; i < order_; ++i) {
                    for (int j = 0; j < order_; ++j) factor(i, j) = Scalar(0);
                    for (int j = 0; j <= i; ++j) {
                        long double value = static_cast<Scalar>(coefficients(i, j));
                        for (int k = 0; k < j; ++k) value -= static_cast<long double>(factor(i, k)) * factor(j, k);
                        fdapde_strong_assert(
                          j != i || (value > 0 && std::isfinite(value)), std::domain_error,
                          "SPD cache: Cholesky pivot must be positive and finite");
                        factor(i, j) = static_cast<Scalar>(i == j ? std::sqrt(value) : value / factor(j, j));
                        fdapde_strong_assert(
                          std::isfinite(factor(i, j)), std::domain_error, "SPD cache: nonfinite Cholesky factor");
                    }
                }
            }
            if constexpr (spd_cache_has_v<Missing, Cache::LogCholesky>) {
                const Scalar inverse_root_two = Scalar(1) / std::sqrt(Scalar(2));
                for (int i = 0; i < order_; ++i)
                    for (int j = 0; j <= i; ++j) {
                        const Scalar value = i == j ? std::log(factor(i, i)) : factor(i, j) * inverse_root_two;
                        fdapde_strong_assert(
                          std::isfinite(value), std::domain_error, "SPD cache: nonfinite log-Cholesky coordinates");
                        data_[offset_<Cache::LogCholesky>() + std::size_t(i) * (i + 1) / 2 + j] = value;
                    }
            }
        }
    }
   private:
    /// @brief copies one common packed quantity without recomputing its spectral reconstruction
    template <typename Quantity, typename Source> void copy_matrix_(const Source& source) {
        const auto matrix = source.template matrix<Quantity>();
        std::copy_n(matrix.data(), std::size_t(order_) * (order_ + 1) / 2, data_ + offset_<Quantity>());
    }
    /// @brief locates a selected quantity after the preceding packed quantities
    template <typename Quantity> std::size_t offset_() const {
        const auto n = static_cast<std::size_t>(order_);
        return (Has<Cache::Spectral> ? n * (n + 1) : 0) +
               ((Quantity::Flags > 2 && Has<Cache::Log>)+(Quantity::Flags > 4 && Has<Cache::Sqrt>)+(
                 Quantity::Flags > 8 && Has<Cache::InverseSqrt>)) *
                 n * (n + 1) / 2 +
               ((Quantity::Flags > 16 &&
                 Has<Cache::LogDividedDifferences>)+(Quantity::Flags > 32 && Has<Cache::Cholesky>)) *
                 n * n;
    }
    /// @brief writes the unit diagonal into one already zeroed packed quantity
    template <typename Quantity> void set_identity_matrix_() {
        auto* target = data_ + offset_<Quantity>();
        for (int i = 0; i < order_; ++i) target[std::size_t(i) * (i + 1) / 2 + i] = Scalar(1);
    }
    /// @brief reconstructs a selected finite packed quantity in the certified eigenvector basis
    template <typename Quantity, typename Spectral, typename Function>
    void prepare_matrix_(const Spectral& spectral, Function function) {
        const auto vectors = spectral.eigenvectors();
        const auto& values = spectral.eigenvalues();
        Vector<Scalar, Rows> transformed;
        if constexpr (Rows == Dynamic) transformed.resize(order_);
        for (int k = 0; k < order_; ++k) transformed[k] = function(values[k]);
        auto* target = data_ + offset_<Quantity>();
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                Scalar value = Scalar(0);
                for (int k = 0; k < order_; ++k) value += vectors(i, k) * transformed[k] * vectors(j, k);
                fdapde_strong_assert(std::isfinite(value), std::domain_error, "SPD cache: nonfinite result");
                target[std::size_t(i) * (i + 1) / 2 + j] = value;
            }
        }
    }
    Scalar* data_;
    int order_;
};

/// @brief owns one independently allocated cache with a slot bound to its selected scalar storage
template <typename Scalar, int Order, typename Policy> struct owning_spd_cache {
    using Slot = spd_cache_slot<Scalar, Order, Policy>;
    std::vector<Scalar> values;
    Slot slot;
    /// @brief allocates precisely the scalar count required by the selected policy
    explicit owning_spd_cache(int n) : values(Slot::scalar_count(n)), slot(values.data(), n) { }
    /// @brief deep-copies retained quantities and rebuilds the independent slot binding
    owning_spd_cache(const owning_spd_cache& other) : values(other.values), slot(values.data(), other.slot.rows()) { }
    /// @brief forbids memberwise assignment that would leave the slot bound to a different buffer
    owning_spd_cache& operator=(const owning_spd_cache&) = delete;
};

}   // namespace internals
}   // namespace fdapde
#endif   // __FDAPDE_LINALG_SPD_CACHE_H__
