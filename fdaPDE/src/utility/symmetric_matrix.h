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

#ifndef __FDAPDE_SYMMETRIC_MATRIX_BASE_H__
#define __FDAPDE_SYMMETRIC_MATRIX_BASE_H__

#include "header_check.h"
#include "matrix_base.h"

namespace fdapde {

template <typename Scalar_, int Size_, int NestAsRefBit_ = 1>
class SymmetricMatrix : public SquareMatrixBase<Size_, SymmetricMatrix<Scalar_, Size_, NestAsRefBit_>> {

public:
    using Base = SquareMatrixBase<Size_, SymmetricMatrix<Scalar_, Size_, NestAsRefBit_>>;
    using Scalar = Scalar_;
    static constexpr int Size = Size_;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int StorageSize = Size_ * (Size_ + 1) / 2;
    static constexpr int NestAsRefBit = NestAsRefBit_;
    static constexpr int XprBits = int(matrix_flags::square) | int(matrix_flags::symmetric);

    // constructors
    constexpr SymmetricMatrix() : data_() {};
    constexpr SymmetricMatrix(const SymmetricMatrix& other) = default;
    constexpr explicit SymmetricMatrix(const std::array<Scalar, StorageSize>& data) : data_(data) { }
    constexpr explicit SymmetricMatrix(const std::vector<Scalar>& data) : data_() {
        fdapde_constexpr_assert(data.size() == StorageSize);
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = data[id];
        }
    }
    template <typename Derived>
    constexpr SymmetricMatrix(const MatrixBase<Rows, Cols, Derived>& xpr) : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_TYPES_CONVERSION_BETWEEN_MATRICES);
        fdapde_static_assert(
          Derived::Rows == Rows && Derived::Cols == Cols,
          YOU_ARE_TRYING_TO_CONSTRUCT_A_MATRIX_WITH_ANOTHER_MATRIX_OF_DIFFERENT_SIZE);
        for (int id = 0; id < StorageSize; ++id) {
            auto[i, j] = inv_index(id);
            data_[id] = xpr.derived().operator()(i, j);
        }
    }
    template <typename Callable>
    constexpr explicit SymmetricMatrix(Callable callable)
        requires(std::is_invocable_v<Callable>)
        : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {
            callable})::result_type FDAPDE_COMMA std::array<Scalar FDAPDE_COMMA StorageSize>>,
          CALLABLE_DOES_NOT_RETURN_SOMETHING_CONVERTIBLE_TO_AN_ARRAY_OF_SCALAR);
        data_ = callable();
    }
    constexpr explicit SymmetricMatrix(const Scalar_ (&data)[StorageSize]) : data_() {
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = data[id];
        }
    }

    // static constructors
    static constexpr SymmetricMatrix<Scalar, Size> Constant(Scalar c) {
        std::array<Scalar, StorageSize> data;
        data.fill(c);
        SymmetricMatrix<Scalar, Size> m(data);
        return m;
    }
    static constexpr SymmetricMatrix<Scalar, Size> Zero() { return Constant(Scalar(0)); }
    static constexpr SymmetricMatrix<Scalar, Size> Ones() { return Constant(Scalar(1)); }
    static constexpr SymmetricMatrix<Scalar, Size> NaN() { return Constant(std::numeric_limits<Scalar>::quiet_NaN()); }
    // copy operator
    constexpr SymmetricMatrix& operator=(const SymmetricMatrix& other) {
        for (int id = 0; id < StorageSize; ++id) { data_[id] = other.data()[id]; }
        return *this;
    };
    // const access
    constexpr Scalar operator()(int i, int j) const{
        assert(i >= 0 && i < Size && j >= 0 && j < Size);
        return (j >= i) ? data_[index(i, j)] : data_[index(j, i)];
    }
    // mutable access
    constexpr Scalar& operator()(int i, int j){
        assert(i >= 0 && i < Size && j >= 0 && j < Size);
        return (j >= i) ? data_[index(i, j)] : data_[index(j, i)];
    }
    // data
    constexpr const Scalar* data() const { return data_.data(); }
    Scalar* data() { return data_.data(); }
    // assignment operator
    // TODO: assignment operator from a generic SymmetricMatrix
    // TODO: assignment operator from a generic SquareMatrix
    // assignment from std::array
    constexpr SymmetricMatrix<Scalar, Rows, NestAsRefBit>& operator=(const std::array<Scalar, StorageSize>& rhs) {
        for (int id = 0; id < StorageSize; ++id) {
            data_[id] = rhs[id];
        }
        return *this;
    }
    constexpr void setConstant(Scalar c) {
        for (int id = 0; id < StorageSize; ++id) data_[id] = c;
    }
    constexpr void setZero() { setConstant(Scalar(0)); }
    constexpr void setOnes() { setConstant(Scalar(1)); }
    // convert to full matrix
    constexpr fdapde::Matrix<Scalar_, Size, Size> full() const {
        fdapde::Matrix<Scalar_, Size, Size> M;
        for (int i = 0; i < Size; ++i) {
            for (int j = i; j < Size; ++j) {
                M(i, j) = (*this)(i, j);
                M(j, i) = (*this)(i, j);
            }
        }
        return M;
    }
    // getters
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }

private:
    std::array<Scalar, StorageSize> data_;

    static constexpr int index(int i, int j){
        assert(j >= i);
        // assumes j >= i
        return i * (2 * Size - i + 1) / 2 + (j - i);
    }
    static constexpr std::pair<int, int> inv_index(int idx){
        int i = 0;
        int offset = 0;

        // loop to find the row i such that idx is within its block
        while (i < Size) {
            int row_len = Size - i;
            if (idx < offset + row_len)
                break;
            offset += row_len;
            ++i;
        }

        int j = i + (idx - offset);
        return {i, j};
    }
};

}

#endif // __FDAPDE_SYMMETRIC_MATRIX_H__