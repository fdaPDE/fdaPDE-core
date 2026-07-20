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

#ifndef __FDAPDE_LINALG_NATIVE_MATRIX_BATCH_H__
#define __FDAPDE_LINALG_NATIVE_MATRIX_BATCH_H__

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "header_check.h"

namespace fdapde::linalg {

template <typename Scalar_, int Rows_, int Cols_> class MatrixBatchView {
    fdapde_static_assert(Rows_ > 0 && Cols_ > 0, MATRIX_BATCH_REQUIRES_FIXED_POSITIVE_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      std::int64_t(Rows_) * std::int64_t(Cols_) <= std::numeric_limits<int>::max(),
      MATRIX_BATCH_MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
    fdapde_static_assert(!std::is_volatile_v<Scalar_>, MATRIX_BATCH_DOES_NOT_SUPPORT_VOLATILE_SCALARS);
    fdapde_static_assert(
      !std::is_same_v<std::remove_cv_t<Scalar_> FDAPDE_COMMA bool>, MATRIX_BATCH_DOES_NOT_SUPPORT_BOOLEAN_SCALARS);

    using ValueType = std::remove_const_t<Scalar_>;
    using MatrixViewType = MatrixView<Scalar_, Rows_, Cols_, RowMajor>;
    using ConstMatrixViewType = MatrixView<const ValueType, Rows_, Cols_, RowMajor>;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = RowMajor;
    static constexpr std::size_t MatrixSize = static_cast<std::size_t>(Rows_) * static_cast<std::size_t>(Cols_);

    MatrixBatchView() = delete;
    template <typename SourceScalar_, std::size_t Extent_>
        requires(std::is_convertible_v<SourceScalar_ (*)[] FDAPDE_COMMA Scalar_ (*)[]>)
    constexpr explicit MatrixBatchView(std::span<SourceScalar_, Extent_> data) : data_(data) {
        if (data_.size() % MatrixSize != 0) {
            throw std::invalid_argument("MatrixBatchView: buffer size is not divisible by the matrix size");
        }
        if (data_.size() / MatrixSize > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error("MatrixBatchView: matrix count exceeds supported range");
        }
    }

    [[nodiscard]] constexpr int size() const { return static_cast<int>(data_.size() / MatrixSize); }
    [[nodiscard]] constexpr bool empty() const { return data_.empty(); }

    constexpr MatrixViewType operator[](int index) &
        requires(!std::is_const_v<Scalar_>)
    {
        return MatrixViewType(data_.data() + offset_(index));
    }
    constexpr ConstMatrixViewType operator[](int index) const& {
        return ConstMatrixViewType(data_.data() + offset_(index));
    }
    constexpr void operator[](int) && = delete;
    constexpr void operator[](int) const&& = delete;
   private:
    [[nodiscard]] constexpr std::size_t offset_(int index) const {
        if (index < 0 || index >= size()) { throw std::out_of_range("MatrixBatchView: matrix index out of range"); }
        return static_cast<std::size_t>(index) * MatrixSize;
    }

    std::span<Scalar_> data_;
};

}   // namespace fdapde::linalg

#endif   // __FDAPDE_LINALG_NATIVE_MATRIX_BATCH_H__
