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

#include <memory>
#include <vector>

#include "cache_policy.h"
#include "header_check.h"

namespace fdapde {

/// @brief computes eigenpairs of a symmetric expression
template <typename XprType> class EVD;

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
    static constexpr int Rows = Order;
    static_assert((Policy::Flags & ~Cache::Spectral::Flags) == 0, "symmetric cache supports None and Spectral only");
    /// @brief binds unused storage without computing any eigenpairs
    symmetric_cache_slot(Scalar* data, int n) : data_(data), n_(n) { }
    /// @brief returns the matrix order associated with the spectral buffer
    int rows() const { return n_; }
    /// @brief counts the selected eigenvector and eigenvalue coefficients
    static std::size_t scalar_count(int n) { return Policy::Flags ? std::size_t(n) * (n + 1) : 0; }
    /// @brief marks eigenpairs stale after a coefficient mutation
    void invalidate() { valid_ = false; }
    /// @brief disables reuse while an untracked mutable pointer may still exist
    void expose_mutable_data() {
        valid_ = false;
        exposed_ = true;
    }
    /// @brief reports whether the slot contains prepared eigenpairs
    /// @details prepared eigenpairs are recomputed on every request after mutable raw storage is exposed
    bool valid() const { return valid_; }
    /// @brief copies ready or invalid cache state without changing the binding
    void copy_from(const symmetric_cache_slot& other) {
        fdapde_assert(n_ == other.n_, std::invalid_argument, "symmetric cache: incompatible order");
        if (other.valid_ && !other.exposed_) std::copy_n(other.data_, scalar_count(n_), data_);
        valid_ = other.valid_ && !other.exposed_ && !exposed_;
    }
    /// @brief prepares eigenpairs and reuses them only while every coefficient write remains tracked
    template <typename Xpr> void prepare(const Xpr& value) {
        if (valid_ && !exposed_) return;
        valid_ = false;
        // raw coefficient aliases bypass write validation, so cache preparation validates their current values
        for (int i = 0; i < n_; ++i)
            for (int j = 0; j <= i; ++j)
                fdapde_strong_assert(
                  std::isfinite(static_cast<Scalar>(value(i, j))), std::invalid_argument,
                  "symmetric coefficients must be finite");
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
    bool exposed_ = false;
};

}   // namespace internals
}   // namespace fdapde
#endif
