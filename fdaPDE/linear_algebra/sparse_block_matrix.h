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

#ifndef __SPARSE_BLOCK_MATRIX_H__
#define __SPARSE_BLOCK_MATRIX_H__

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "../utils/assert.h"

namespace fdapde {
namespace core {

// A C++17 Eigen-compatible SparseBlockMatrix implementation (only ColMajor support). Use Eigen naming conventions
template <typename Scalar_, int Rows_, int Cols_, int Options_ = Eigen::ColMajor, typename StorageIndex_ = int>
struct SparseBlockMatrix :
    public Eigen::SparseMatrixBase<SparseBlockMatrix<Scalar_, Rows_, Cols_, Options_, StorageIndex_>> {
    static_assert(Rows_ > 1 || Cols_ > 1, "Rows_>1 || Cols_>1 failed");
    typedef Scalar_ Scalar;
    typedef StorageIndex_ StorageIndex;
    typedef
      typename Eigen::internal::ref_selector<SparseBlockMatrix<Scalar_, Rows_, Cols_, Options_, StorageIndex_>>::type
        Nested;

    // default constructor
    SparseBlockMatrix() = default;

    // initialize from list of matrices
    // constructor enabled only if more than one block is passed
    template <typename... E, typename std::enable_if<(sizeof...(E) > 1), int>::type = 0>
    SparseBlockMatrix(const E&... m) {
        // compile time checks
        static_assert((std::is_same<typename E::Scalar, Scalar_>::value && ...), "blocks with different scalar type");
        static_assert(sizeof...(E) == Rows_ * Cols_, "supplied blocks < Rows_*Cols_");

        // unfold parameter pack and extract size of blocks and overall matrix size
        std::size_t i = 0, j = 0, k = 0;
        (
          [&] {
              // row and column block indexes
              std::size_t r_blk = std::floor(i / Cols_);
              std::size_t c_blk = i % Cols_;

              if (r_blk == 0) {   // take columns dimension from first row
                  cols_ += m.cols();
                  outer_size_[j++] = m.cols();
                  outer_offset_[j] = m.cols() + outer_offset_[j - 1];
              } else if (m.cols() != outer_size_[c_blk])
                  throw std::length_error("blocks have incompatible dimensions");
              if (c_blk == 0) {   // take rows dimension from first column
                  rows_ += m.rows();
                  inner_size_[k++] = m.rows();
                  inner_offset_[k] = m.rows() + inner_offset_[k - 1];
              } else if (m.rows() != inner_size_[r_blk])
                  throw std::length_error("blocks have incompatible dimensions");
              i++;
          }(),
          ...);
        // evaluate each block and store in internal storage
        blocks_.reserve(Rows_ * Cols_);
        ([&] { blocks_.emplace_back(m); }(), ...);
    };

    // read/write access to individual blocks
    const SpMatrix<double>& block(Eigen::Index row, Eigen::Index col) const {
        fdapde_assert(row >= 0 && row < Rows_ && col >= 0 && col < Cols_);
        return blocks_[row * Cols_ + col];
    }
    SpMatrix<double>& block(Eigen::Index row, Eigen::Index col) {
        fdapde_assert(row >= 0 && row < Rows_ && col >= 0 && col < Cols_);
        return blocks_[row * Cols_ + col];
    }

    // provides an estimate of the nonzero elements of the matrix
    Eigen::Index nonZerosEstimate() const {
        if (blocks_.size() == 0) return 0;   // empty matrix
        Eigen::Index nnz = 0;
        for (const auto& b : blocks_) nnz += b.nonZerosEstimate();
        return nnz;
    }

    // number of rows and columns
    inline Eigen::Index rows() const { return rows_; }
    inline Eigen::Index cols() const { return cols_; }
    // number of blocks by rows and by cols
    inline Eigen::Index blockRows() const { return Rows_; }
    inline Eigen::Index blockCols() const { return Cols_; }

    // non-const access to the (i,j)-th element
    Scalar& coeffRef(Eigen::Index row, Eigen::Index col) {
        fdapde_assert(row >= 0 && row < rows_ && col >= 0 && col < cols_);
        // block indexes where (row, col) belongs to
        Eigen::Index b_i = innerBlockIndex(row);
        Eigen::Index b_j = outerBlockIndex(col);
        // (row, col) pair mapped inside block
        Eigen::Index b_inner = indexToBlockInner(row);
        Eigen::Index b_outer = indexToBlockOuter(col);

        return blocks_[b_i * Cols_ + b_j].coeffRef(b_inner, b_outer);
    }

    // const access to of the (i,j)-th element
    Scalar coeff(Eigen::Index row, Eigen::Index col) const {
        fdapde_assert(row >= 0 && row < rows_ && col >= 0 && col < cols_);
        // block indexes where (row, col) belongs to
        Eigen::Index b_i = innerBlockIndex(row);
        Eigen::Index b_j = outerBlockIndex(col);
        // (row, col) pair mapped inside block
        Eigen::Index b_inner = indexToBlockInner(row);
        Eigen::Index b_outer = indexToBlockOuter(col);

        return blocks_[b_i * Cols_ + b_j].coeff(b_inner, b_outer);
    }

    // the outer block index where i belongs to
    inline Eigen::Index outerBlockIndex(Eigen::Index i) const {
        return std::distance(outer_offset_.begin(), std::upper_bound(outer_offset_.begin(), outer_offset_.end(), i)) - 1;
    }
    // the inner block index where i belongs to
    inline Eigen::Index innerBlockIndex(Eigen::Index i) const {
        return std::distance(inner_offset_.begin(), std::upper_bound(inner_offset_.begin(), inner_offset_.end(), i)) - 1;
    }

    // the outer index relative to the block where i belongs to
    inline Eigen::Index indexToBlockOuter(Eigen::Index i) const {
        return i - *(std::upper_bound(outer_offset_.begin(), outer_offset_.end(), i) - 1);
    }
    // the inner index relative to the block where i belongs to
    inline Eigen::Index indexToBlockInner(Eigen::Index i) const {
        return i - *(std::upper_bound(inner_offset_.begin(), inner_offset_.end(), i) - 1);
    }

    inline bool isCompressed() const { return true; }   // matrix in compressed format (blocks are in compressed format)
   protected:
    std::vector<Eigen::SparseMatrix<Scalar>> blocks_ {};
    std::array<std::size_t, Cols_ + 1> outer_offset_ {};   // starting outer index of each block
    std::array<std::size_t, Rows_ + 1> inner_offset_ {};   // starting inner index of each block
    std::array<std::size_t, Cols_> outer_size_ {};         // outer size of each block
    std::array<std::size_t, Rows_> inner_size_ {};         // inner size of each block
    Eigen::Index cols_ = 0, rows_ = 0;                     // matrix dimensions
};

}   // namespace core
}   // namespace fdapde

// definition of proper symbols in Eigen::internal namespace
namespace Eigen {
namespace internal {
// import symbols from fdapde namespace
using fdapde::core::SparseBlockMatrix;

// trait definition
template <typename Scalar_, int Rows_, int Cols_, int Options_, typename StorageIndex_>
struct traits<SparseBlockMatrix<Scalar_, Rows_, Cols_, Options_, StorageIndex_>> {
    typedef Scalar_ Scalar;   // type of stored coefficients
    typedef StorageIndex_ StorageIndex;
    typedef Sparse StorageKind;   // sparse storage
    typedef MatrixXpr XprKind;    // expression type (matrix expression)
    enum {
        // we know the number of blocks at compile time, but the number of rows and cols
        // of the overall matrix is unknown at compile time
        RowsAtCompileTime     = Dynamic,
        ColsAtCompileTime     = Dynamic,
        MaxRowsAtCompileTime  = Dynamic,
        MaxColsAtCompileTime  = Dynamic,
        Flags                 = Options_ |   // inherits supplied stoarge mode, defaulted to ColMajor storage
	                        LvalueBit,   // the expression has a coeffRef() method, i.e. it is writable
        IsVectorAtCompileTime = 0,
        IsColMajor            = Options_ & Eigen::RowMajorBit ? 0 : 1
    };
};

// evaluator definition
template <typename Scalar_, int Rows_, int Cols_, int Options_, typename StorageIndex_>
struct evaluator<SparseBlockMatrix<Scalar_, Rows_, Cols_, Options_, StorageIndex_>> :
    public evaluator_base<SparseBlockMatrix<Scalar_, Rows_, Cols_, Options_, StorageIndex_>> {
    // typedefs expected by eigen internals
    typedef SparseBlockMatrix<Scalar_, Rows_, Cols_, Options_, StorageIndex_> XprType;
    typedef Scalar_ Scalar;
    enum {   // required compile time constants
        CoeffReadCost = NumTraits<Scalar_>::ReadCost,
        Flags         = Options_ | LvalueBit
    };

    // InnerIterator defines the SparseBlockMatrix itself
    class InnerIterator {
       public:
        typedef typename traits<XprType>::Scalar Scalar;
        typedef typename traits<XprType>::StorageIndex StorageIndex;
        typedef typename SparseMatrix<Scalar>::InnerIterator IteratorType;
        // costructor (outer is the index of the column over which we are iterating, for ColMajor storage).
        InnerIterator(const evaluator<XprType>& eval, Index outer) :
            m_mat(eval.xpr_),
            outer_(outer),
            innerBlockIndex(0),
            outerBlockIndex(m_mat.outerBlockIndex(outer)),
            innerOffset(0),
            outerOffset(m_mat.indexToBlockOuter(outer)) {
            inner_ = IteratorType(m_mat.block(0, outerBlockIndex), outerOffset);
            this->operator++();   // init iterator
        };

        InnerIterator& operator++() {
            while (!inner_) {   // current block is over, search for next not-empty block, if any
                if (innerBlockIndex == m_mat.blockRows() - 1) {
                    m_index = -1;
                    return *this;
                }   // end of iterator
                inner_ = IteratorType(m_mat.block(++innerBlockIndex, outerBlockIndex), outerOffset);
                innerOffset += m_mat.block(0, outerBlockIndex).rows();   // increase innerOffset
            }
            m_value = inner_.value();
            m_index = innerOffset + inner_.index();
            ++inner_;
            return *this;
        };
        // access methods
        inline Scalar value() const { return m_value; }         // value pointed by the iterator
        inline Index col() const { return outer_; }             // current column (assume ColMajor order)
        inline Index row() const { return index(); }            // current row (assume ColMajor order)
        inline Index outer() const { return outer_; }           // outer index
        inline StorageIndex index() const { return m_index; }   // inner index
        operator bool() const { return m_index >= 0; }          // false when the iterator is over
       protected:
        IteratorType inner_;    // current block inner iterator
        const XprType& m_mat;   // SparseBlockMatrix to evaluate
        Scalar m_value;         // value pointed by the iterator
        StorageIndex m_index;   // current inner index
        Index outer_;           // outer index as received from the constructor

        // internals
        Index innerBlockIndex, outerBlockIndex;   // indexes of block where iterator is iterating
        Index innerOffset, outerOffset;
    };
    // constructor
    evaluator(const XprType& xpr) : xpr_(xpr) {};
    inline Index nonZerosEstimate() const { return xpr_.nonZerosEstimate(); }

    // SparseBlockMatrix to evaluate
    const SparseBlockMatrix<Scalar_, Rows_, Cols_, Options_, StorageIndex_>& xpr_;
};

}   // namespace internal
}   // namespace Eigen

#endif   // __SPARSE_BLOCK_MATRIX_H__
