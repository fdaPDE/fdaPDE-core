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


#ifndef __FSPAI_H__
#define __FSPAI_H__

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unordered_map>
#include <vector>

#include "../utils/symbols.h"

namespace fdapde {

template <typename T>
concept is_eigen_sparse_matrix = std::is_base_of_v<Eigen::SparseMatrixBase<T>, T>;
  
namespace internals {
  
// a class to represent the sparsity pattern of a matrix
template <typename Index_, int Options_> struct sparsity_pattern {
    using Index = Index_;
    static constexpr int StorageOrder = Options_;   // either RowMajor or ColMajor

    class sparsity_line {
        std::vector<Index> nnzeros_;   // indexes of non-zero entries on this line
        Index size_ = 0;               // the size of the matrix to which this sparsity_line belongs to
       public:
        sparsity_line() noexcept : nnzeros_(), size_(0) { }
        sparsity_line(const sparsity_line&) noexcept = default;
        sparsity_line(sparsity_line&&) noexcept = default;

        sparsity_line(Index size) : nnzeros_(), size_(size) { }
        sparsity_line(Index size, const std::vector<Index>& nnzeros) noexcept : nnzeros_(nnzeros), size_(size) { }
        template <typename Iterator>
            requires(std::is_convertible_v<typename std::iterator_traits<Iterator>::value_type, Index>)
        sparsity_line(Index size, Iterator begin, Iterator end) : nnzeros_(begin, end), size_(size) { }
        template <typename... Args>
            requires((std::is_convertible_v<Args, Index>) && ...)
        sparsity_line(Index size, const Args&... args) : nnzeros_(), size_(size) {
            ([&]() { nnzeros_.push_back(static_cast<Index>(args)); }(), ...);
        }
        // accessors
        Index nnzeros() const { return nnzeros_.size(); }
        typename std::vector<Index>::const_iterator begin() const { return nnzeros_.begin(); }
        typename std::vector<Index>::const_iterator end() const { return nnzeros_.end(); }
        Index operator[](Index i) const {   // access the i-th nonzero of the line
            fdapde_assert(static_cast<std::size_t>(i) < nnzeros_.size());
            return nnzeros_[i];
        }
        Index size() const { return size_; }
        bool has_nnzero_at(Index pos) const { return std::find(nnzeros_.begin(), nnzeros_.end(), pos); }
        bool empty() const { return nnzeros_.size() == 0; }
        typename std::vector<Index>::const_iterator find(Index pos) const {
            fdapde_assert(pos < size_);
            return std::find(nnzeros_.begin(), nnzeros_.end(), pos);
        }
        // modifiers
        void nnzero_insert_unique(Index pos) {   // ammortized O(log(n)) insertion with uniqueness guarantees
            fdapde_assert(pos < size_);
            if (std::upper_bound(nnzeros_.begin(), nnzeros_.end(), pos) == nnzeros_.end()) { nnzeros_.push_back(pos); }
            return;
        }
        void nnzero_insert(Index pos) {
            fdapde_assert(pos < size_);
            nnzeros_.push_back(pos);
        }
        void resize(Index size) { size_ = size; }
        void erase(Index idx) {
            auto it = std::find(nnzeros_.begin(), nnzeros_.end(), idx);
            if (it != nnzeros_.end()) { nnzeros_.erase(it); }
        }
    };  
    using iterator = typename std::vector<sparsity_line>::iterator;
    using value_type = sparsity_line;

    sparsity_pattern() noexcept : inner_size_(0), outer_size_(0), sparsity_() { }
    sparsity_pattern(const sparsity_pattern&) noexcept = default;
    sparsity_pattern(sparsity_pattern&&) noexcept = default;

    sparsity_pattern(int rows, int cols) noexcept :
        inner_size_(StorageOrder == RowMajor ? cols : rows),
        outer_size_(StorageOrder == RowMajor ? rows : cols),
        sparsity_(outer_size_) {
        for (auto& line : sparsity_) line.resize(inner_size_);
    }
    template <typename MatrixType>
        requires(fdapde::is_eigen_sparse_matrix<MatrixType> &&
                 requires(MatrixType m) { typename MatrixType::InnerIterator; } &&
                 ((MatrixType::IsRowMajor == RowMajor && StorageOrder == RowMajor) || StorageOrder == ColMajor))
    sparsity_pattern(const MatrixType& m) :
        inner_size_(StorageOrder == RowMajor ? m.cols() : m.rows()),
        outer_size_(StorageOrder == RowMajor ? m.rows() : m.cols()),
        sparsity_(outer_size_) {
        for (auto& line : sparsity_) line.resize(inner_size_);
        for (int k = 0; k < m.outerSize(); ++k) {
            for (typename MatrixType::InnerIterator it(m, k); it; ++it) { sparsity_[it.row()].nnzero_insert(it.col()); }
        }
    }
    // accessors
    const value_type& operator[](Index i) const {
        fdapde_assert(i < outer_size_);
        return sparsity_[i];
    }
    value_type& operator[](Index i) {
        fdapde_assert(i < outer_size_);
        return sparsity_[i];
    }
    const value_type& operator()(Index i, Index j) const { return sparsity_[i][j]; }
    value_type& operator()(Index i, Index j) { return sparsity_[i][j]; }
    Index rows() const { return StorageOrder == RowMajor ? outer_size_ : inner_size_; }
    Index cols() const { return StorageOrder == RowMajor ? inner_size_ : outer_size_; }
    Index inner_size() const { return inner_size_; }
    Index outer_size() const { return outer_size_; }
    // modifiers
    void resize(Index rows, Index cols) {
        inner_size_ = StorageOrder == RowMajor ? cols : rows;
        outer_size_ = StorageOrder == RowMajor ? rows : cols;
        sparsity_.resize(outer_size_);
        for (auto& line : sparsity_) line.resize(inner_size_);
        return;
    }
    void resize(Index size) { resize(size, size); }
   private:
    Index inner_size_ = 0, outer_size_ = 0;
    std::vector<sparsity_line> sparsity_;   // guarantees O(1) line access
};

}   // namespace internals

// implementation of the Factorized Sparse Approximate Inverse algorithm with sparsity pattern update, for the sparse
// SPD square approximation of the inverse of a sparse SPD square matrix
template <typename MatrixType_>
    requires(fdapde::is_eigen_sparse_matrix<MatrixType_>)
struct FSPAI {
    using MatrixType = MatrixType_;
    using Index = Eigen::Index;
    using StorageIndex = typename MatrixType::StorageIndex;
    using Scalar = typename MatrixType::Scalar;
    using SparseMatrixType = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
    using DenseMatrixType  = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVectorType  = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using CholeskySolver   = Eigen::LLT<DenseMatrixType>;
    using MatrixL = Eigen::TriangularView<const SparseMatrixType, Eigen::Lower>;
    using MatrixU = Eigen::TriangularView<const typename SparseMatrixType::ConstTransposeReturnType, Eigen::Upper>;
   private:
    SparseMatrixType L_;       // the sparse approximate inverse of the Cholesky factor of A_

    // parameters
    int alpha_ = 10;           // number of sparsity pattern updates for each column of matrix
    int beta_ = 10;            // maximum number of inserted indices to the sparsity pattern
    double epsilon_ = 0.005;   // K-condition number tolerance treshold for sparsity pattern update

    void compute_impl_(const MatrixType& matrix, int alpha, int beta, double epsilon) {
        std::vector<Eigen::Triplet<double>> tripet_list;
        int n_cols = matrix.cols();
        internals::sparsity_pattern<Index, ColMajor> matrix_sparsity_(matrix);   // compute input sparsity pattern
        L_.resize(n_cols, n_cols);
        CholeskySolver solver;
        DenseMatrixType Ak;            // memory buffer for incremental build of matrix A(Jk_, Jk_)
        DenseVectorType bk;            // memory buffer for incremental build of rhs vector A(Jk_, k)
        DenseVectorType Lk_(n_cols);   // the k-th column of the approximate inverse of L_

        // cycle over each column of the sparse matrix A_
        for (Index k = 0; k < n_cols; ++k) {
            std::vector<Index> Ck_ {};         // indexes eligible to enter in the sparsity pattern of column k
            std::vector<Index> Jk_ {};         // sparsity patterns of column k
            std::vector<Index> delta_Jk_ {};   // indexes added to the sparsity pattern of column k at iteration s - 1

            Lk_.fill(0);
            Lk_[k] = 1.0 / (std::sqrt(matrix.coeff(k, k)));
            Jk_.push_back(k);
            delta_Jk_.push_back(k);

            int Jk_offset = 1;
            // perform alpha_ steps of approximate inverse update along column k
            for (Index s = 0; s < alpha && !delta_Jk_.empty(); ++s) {
                if (s != 0) {
                    // build SPD linear system A(Jk_, Jk_)*yk = A(Jk_, k), being k fixed. Jk_ is incremental, query the
                    // input matrix only for those entries which have entered the sparsity pattern at iteration s - 1
                    Index nnzeros_ = Jk_.size();
                    Ak.conservativeResize(nnzeros_ - 1, nnzeros_ - 1);
                    bk.conservativeResize(nnzeros_ - 1);
                    for (int i = Jk_offset; i < nnzeros_; ++i) {
                        for (int j = 1; j < i + 1; ++j) {   // build only lower-triangular part, then symmetrize
                            Ak(i - 1, j - 1) = matrix.coeff(Jk_[i], Jk_[j]);
                        }
                        bk[i - 1] = matrix.coeff(Jk_[i], k);
                    }
                    Jk_offset = nnzeros_;
                    solver.compute(Ak.template selfadjointView<Eigen::Lower>());
                    DenseVectorType yk = solver.solve(bk);
                    // update approximate inverse
                    Scalar l_kk = 1.0 / (std::sqrt(matrix.coeff(k, k) - bk.dot(yk)));
                    Lk_[k] = l_kk;                                                            // diagonal entry
                    for (int i = 1; i < nnzeros_; ++i) { Lk_[Jk_[i]] = -l_kk * yk[i - 1]; }   // off-diagonal entries
                }

                // sparsity pattern update (select entry which improves the K condition number better than average)
                for (auto row = delta_Jk_.begin(); row != delta_Jk_.end(); ++row) {
                    for (auto j : matrix_sparsity_[*row]) {
                        // Cholesky factor is upper triangular
                        if (j > k && std::find(Ck_.begin(), Ck_.end(), j) == Ck_.end()) { Ck_.push_back(j); }
                    }
                }
                delta_Jk_.clear();
                std::unordered_map<Index, double> hatJk_ {};
                for (Index j : Ck_) {
                    if (std::find(Jk_.begin(), Jk_.end(), j) == Jk_.end()) {   // nonzero entry not found at (j, k)
                        Scalar v = 0;
                        // Compute A(j, Jk) * Lk(Jk) (considering symmetry)
                        for (Index l : Jk_) {
                            v += (k != j) ? 2 * matrix.coeff(j, l) * Lk_[l] : matrix.coeff(j, l) * Lk_[l];
                        }
                        hatJk_.emplace(j, v);
                    }
                }
                Scalar tau_k = 0;     // average improvement to the K-condition number
                Scalar max_tau = 0;   // best improvement to the K-condition number
                for (auto& [j, v] : hatJk_) {
                    Scalar tau_jk = v * v / matrix.coeff(j, j);
                    hatJk_[j] = tau_jk;
                    // update average and maximum value
                    tau_k += tau_jk;
                    if (tau_jk > max_tau) max_tau = tau_jk;
                }
                if (max_tau > epsilon) {
                    tau_k /= hatJk_.size();
                    // select most promising first beta_ entries according to average heuristic
                    for (Index idx = 0; idx < beta && !hatJk_.empty(); ++idx) {
                        auto it = std::max_element(hatJk_.begin(), hatJk_.end(), [](const auto& p1, const auto& p2) {
                            return p1.second < p2.second;
                        });
                        // sparsity pattern update
                        if (it->second >= tau_k) {
                            Jk_.push_back(it->first);
                            delta_Jk_.push_back(it->first);
                            hatJk_.erase(it);
                        } else {   // if optimal element is not best than tau_k, no hope to find a better one
                            break;
                        }
                    }
                }
            }
            // store column k-th result
            for (Index i : Jk_) { tripet_list.emplace_back(i, k, Lk_[i]); }
        }
        // build final result
        L_.setFromTriplets(tripet_list.begin(), tripet_list.end());
        return;
    }
  
   public:
    // constructor
    FSPAI() noexcept = default;
    FSPAI(const FSPAI&) noexcept = default;
    FSPAI(FSPAI&&) noexcept = default;

    FSPAI(const MatrixType& matrix) noexcept : L_() { compute(matrix); }
    FSPAI(const MatrixType& matrix, int alpha, int beta, double epsilon) noexcept :
        L_(), alpha_(alpha), beta_(beta), epsilon_(epsilon) {
        compute(matrix, alpha_, beta_, epsilon_);
    }

    // computes an approximation of the Cholesky factor of matrix
    void compute(const MatrixType& matrix) {
        fdapde_assert(matrix.rows() == matrix.cols() && matrix.rows() > 0 && matrix.cols() > 0);
        compute_impl_(matrix, alpha_, beta_, epsilon_);   // use defaults      
    }
    void compute(const MatrixType& matrix, int alpha, int beta, double epsilon) {
        fdapde_assert(matrix.rows() == matrix.cols() && matrix.rows() > 0 && matrix.cols() > 0);
        compute_impl_(matrix, alpha, beta, epsilon);
    }
    // accessors
    Index rows() const { L_.rows(); }
    Index cols() const { L_.cols(); }
    MatrixL getL() const { return MatrixL(L_); }   // the Cholesky factor of the approximate inverse of matrix
    MatrixU getU() const { return MatrixU(L_.transpose()); }
    SparseMatrixType inverse() const { return getL() * getU(); }   // the factorized sparse approximate inverse

    // linear system solve
    template <typename Other> void solveInPlace(const Eigen::MatrixBase<Other>& other) const {
        fdapde_assert(L_.rows() == other.rows());
        MatrixL(L_).solveInPlace(other);
    }
    template <typename Other> DVector<Scalar> solve(const Eigen::MatrixBase<Other>& other) const {
        fdapde_assert(L_.rows() == other.rows());
        return MatrixL(L_).solve(other);
    }
};

}   // namespace fdapde

#endif   // __FSPAI_H__
