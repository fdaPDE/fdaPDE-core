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

#ifndef __RBKI_H__
#define __RBKI_H__

namespace fdapde {

// randomized SVD decomposition of a matrix, as described in
// J. A. Tropp and R. J. Webber. (2023) Randomized algorithms for low-rank matrix approximation: Design, analysis, and
// applications, Alg 5.4, pag 18.
template <typename MatrixType>
    requires(internals::is_eigen_dense_xpr_v<MatrixType>)
class RBKI {
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using qr_t     = Eigen::HouseholderQR<matrix_t>;
    using svd_t    = Eigen::JacobiSVD<matrix_t>;
   public:

    RBKI() noexcept =default;
    RBKI(const MatrixType& m, int rank) : tol_(1e-5), max_iter_(50), seed_(std::random_device()()) { compute(m, rank); }
    RBKI(const MatrixType& m, int rank, double tol, int max_iter, int seed = random_seed) :
        tol_(tol), max_iter_(max_iter), seed_(seed == random_seed ? std::random_device()() : seed) {
        compute(m, rank);
    }
    RBKI(double tol, int max_iter, int seed = random_seed) :
        tol_(tol), max_iter_(max_iter), seed_(seed == random_seed ? std::random_device()() : seed) { }

    // computes the decomposition
    void compute(const MatrixType& A, int rank) {
        bool transposed = A.rows() > A.cols();
        int block_sz = ((transposed ? A.cols() : A.rows()) <= 100) ? 1 : 10;
        transposed ? compute_(A.transpose(), rank, block_sz) : compute(A, rank, block_sz);
        return;
    }
    void compute(const MatrixType& A, int rank, int block_sz) {
        A.rows() > A.cols() ? compute_(A.transpose(), rank, block_sz) : compute(A, rank, block_sz);
        return;
    }
    // observers
    const matrix_t& matrixU() const { return U_; }
    const matrix_t& matrixV() const { return V_; }
    const vector_t& singularValues() const { return Sigma_; }
    int rank() const { return rank_; }
   private:
    template <typename MatrixType_>
        requires(internals::is_eigen_dense_xpr_v<MatrixType_>)
    void compute_(const MatrixType_& A, int rank, int block_sz) {
        int rows = A.rows();
        int cols = A.cols();
        // adjust maximum iteration number and maximum Krylov Subspace dimension
        int max_iter = std::min(max_iter_, std::min(rows, cols) / block_sz + 1);
        int max_dim = (max_iter + 1) * block_sz;

        // approximate range of A
        matrix_t Omega = internals::gaussian_matrix(cols, block_sz, 1.0, seed_);
        matrix_t Q(rows, max_dim);
        matrix_t B(cols, max_dim);
        Q.leftCols(block_sz) = A * Omega;
        qr_t qr(Q.leftCols(block_sz));
        Q.leftCols(block_sz) = qr.householderQ() * matrix_t::Identity(rows, block_sz);
        B.leftCols(block_sz) = A.transpose() * Q.leftCols(block_sz);
        // Krylov subspace iteration loop initialization
        int i = 0;
        svd_t svd(B.leftCols(block_sz).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        int n = std::min(rank, block_sz);
        int m = block_sz;
        matrix_t Ar = A * svd.matrixV().leftCols(n);
        matrix_t E = Ar - Q.leftCols(block_sz) * svd.matrixU().leftCols(n) * svd.singularValues().head(n).asDiagonal();
        double res_err = E.colwise().template lpNorm<2>().maxCoeff();
        // compute Krylov subspace [A * \Omega, (A * A^\top) * A * \Omega, ..., (A * A^\top)^{q} * A * \Omega]
        for (; res_err > tol_ && i < max_iter; i++) {
            // krylov subspace update
            Q.middleCols((i + 1) * block_sz, block_sz) = A * B.middleCols(i * block_sz, block_sz);
            Q.middleCols((i + 1) * block_sz, block_sz) =
              BCGS_(Q.leftCols((i + 1) * block_sz), Q.middleCols((i + 1) * block_sz, block_sz));
            B.middleCols((i + 1) * block_sz, block_sz) = A.transpose() * Q.middleCols((i + 1) * block_sz, block_sz);
            // residual error update
            svd.compute(B.leftCols((i + 2) * block_sz).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            int n = std::min(rank, (i + 2) * block_sz);
            E = A * svd.matrixV().leftCols(n) -
                Q.leftCols((i + 2) * block_sz) * svd.matrixU().leftCols(n) * svd.singularValues().head(n).asDiagonal();
            res_err = E.colwise().template lpNorm<2>().maxCoeff();
            m += block_sz;
        }
        // store result
        rank = fdapde::min(svd.singularValues().size(), rank);
        rank_ = rank;
        Sigma_ = svd.singularValues().head(rank);
        U_ = Q.leftCols(m) * svd.matrixU().leftCols(rank);
        V_ = svd.matrixV().leftCols(rank);
        return;
    }

    // for X = [X_1 ... X_n], X_i \in R^{n, m} and y \in R^{n, m}, performs a Block Classical Gram-Schmidt step as
    // y = y - \sum_{j} (X_j * X_j^\top) * y
    template <typename Lhs_, typename Rhs_> matrix_t BCGS_(const Lhs_& X, const Rhs_& y) {
        int rows = y.rows();
        int cols = y.cols();
        qr_t qr;
        matrix_t proj = (matrix_t::Identity(rows, rows) - X * X.transpose()); // (I - X * X^\top)

        matrix_t orth_block = proj * y;   // (I - X * X^\top) * y = y - \sum_{j} (X_j * X_j^\top) * y
        orth_block = proj * orth_block;   // repeat orthogonalization (stabilize)

	// perform final stabilzed QR
        qr.compute(orth_block);
        orth_block = qr.householderQ() * matrix_t::Identity(rows, cols);
        return orth_block;
    }

    double tol_ = 1e-5;
    int max_iter_ = 50;
    int seed_;

    matrix_t U_, V_;
    vector_t Sigma_;
    int rank_;
};

// randomized eigenvalue decomposition of a (SPD) matrix, as described in
// J. A. Tropp and R. J. Webber. (2023) Randomized algorithms for low-rank matrix approximation: Design, analysis, and
// applications, Alg 5.8, pag 22.
template <typename MatrixType>
    requires(internals::is_eigen_dense_xpr_v<MatrixType>)
class NysRBKI {
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using qr_t     = Eigen::HouseholderQR<matrix_t>;
    using chol_t   = Eigen::LLT<matrix_t>;
    using svd_t    = Eigen::JacobiSVD<matrix_t>;
   public:

    NysRBKI() noexcept =default;
    NysRBKI(const MatrixType& m, int rank) noexcept : tol_(1e-5), max_iter_(50), seed_(std::random_device()()) {
        compute(m, rank);
    }
    NysRBKI(const MatrixType& m, int rank, double tol, int max_iter, int seed = random_seed) noexcept :
        tol_(tol), max_iter_(max_iter), seed_(seed == random_seed ? std::random_device()() : seed) {
        compute(m, rank);
    }
    NysRBKI(double tol, int max_iter, int seed = random_seed) noexcept :
        tol_(tol), max_iter_(max_iter), seed_(seed == random_seed ? std::random_device()() : seed) { }

    // computes the decomposition
    void compute(const MatrixType& A, int rank) { compute(A, rank, (A.rows() <= 100) ? 1 : 10); }
    void compute(const MatrixType& A, int rank, int block_sz) {
        rank_ = rank;
        int rows = A.rows();
        int cols = A.cols();
        // adjust maximum iteration number and maximum Krylov Subspace dimension
        int max_iter = std::min(max_iter_, std::min(rows, cols) / block_sz + 1);
        int max_dim = (max_iter + 1) * block_sz;
        double shift = A.diagonal().sum() * std::numeric_limits<double>::epsilon();   // epsilon_shift

        // Krylov subspace iteration loop initialization
        matrix_t X(rows, max_dim), Y(rows, max_dim);
        qr_t qr(internals::gaussian_matrix(rows, block_sz, 1.0, seed_));
        X.leftCols(block_sz) = qr.householderQ() * matrix_t::Identity(rows, block_sz);
        Y.leftCols(block_sz) = A * X.leftCols(block_sz);
        matrix_t S = matrix_t::Zero(max_dim, max_dim);   // matrix [X_0, ..., X_{i - 1}]^\top * [Y_1, ..., Y_{i}]
        svd_t svd;
        int i = 0;
        double res_err = std::numeric_limits<double>::max();
        // compute Krylov subspace [A * \Omega, (A * A^\top) * A * \Omega, ..., (A * A^\top)^{q} * A * \Omega]
        for (; res_err > tol_ && i < max_iter; i++) {
            // krylov subspace update
            X.middleCols((i + 1) * block_sz, block_sz) =
              Y.middleCols(i * block_sz, block_sz) + shift * X.middleCols(i * block_sz, block_sz);
            // incremental update of [X_0, ..., X_{i - 1}]^\top * [Y_1, ..., Y_{i}]
            {
                matrix_t tmp = matrix_t::Zero(X.rows(), (i + 1) * block_sz);
                tmp.middleCols(std::max(i - 1, 0) * block_sz, block_sz) =
                  X.middleCols(std::max(i - 1, 0) * block_sz, block_sz);
                tmp.middleCols(i * block_sz, block_sz) = X.middleCols(i * block_sz, block_sz);
                S.block(0, i * block_sz, (i + 1) * block_sz, block_sz) =
                  tmp.transpose() * X.middleCols((i + 1) * block_sz, block_sz);
            }
            // subspace orthogonalization
            auto [Q, R] = BCGS_(X.leftCols((i + 1) * block_sz), X.middleCols((i + 1) * block_sz, block_sz));
            X.middleCols((i + 1) * block_sz, block_sz) = Q;
	    Y.middleCols((i + 1) * block_sz, block_sz) = A * X.middleCols((i + 1) * block_sz, block_sz);
            // Nystrom factor computation
            chol_t chol(S.block(0, 0, (i + 1) * block_sz, (i + 1) * block_sz));
            S.block((i + 1) * block_sz, i * block_sz, block_sz, block_sz) = R;
            matrix_t F = chol.matrixU().solve<Eigen::OnTheRight>(S.block(0, 0, (i + 2) * block_sz, (i + 1) * block_sz));
            // residual error update
            svd.compute(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
            int m = std::min(rank, (i + 1) * block_sz);
            matrix_t E = Y.leftCols((i + 2) * block_sz) * svd.matrixU().leftCols(m) -
                         X.leftCols((i + 2) * block_sz) * svd.matrixU().leftCols(m) *
                           (svd.singularValues().head(m).array().pow(2) - shift).matrix().asDiagonal();
            res_err = std::sqrt(2) * E.colwise().template lpNorm<2>().maxCoeff();
        }
	// store result
        rank = fdapde::min(svd.singularValues().size(), rank);
        U_ = X.leftCols((i + 1) * block_sz) * svd.matrixU().leftCols(rank);
        Lambda_ = (svd.singularValues().head(rank).array().pow(2) - shift).matrix();
        for (int i = 0; i < Lambda_.rows(); ++i) {   // set to zero possible negative eigenvalues due to epsilon shift
            if (Lambda_[i] < 0) Lambda_[i] = 0;
        }
        return;
    }
    // observers
    const matrix_t& matrixU() const { return U_; }
    const vector_t& eigenValues() const { return Lambda_; }
    int rank() const { return rank_; }
   private:
    // for X = [X_1 ... X_n], X_i \in R^{n, m} and y \in R^{n, m}, performs a Block Classical Gram-Schmidt step as
    // y = y - \sum_{j} (X_j * X_j^\top) * y
    template <typename Lhs_, typename Rhs_> std::pair<matrix_t, matrix_t> BCGS_(const Lhs_& X, const Rhs_& y) {
        int rows = y.rows();
        int cols = y.cols();
        qr_t qr;
        matrix_t proj = (matrix_t::Identity(rows, rows) - X * X.transpose()); // (I - X * X^\top)

        matrix_t orth_block = proj * y;   // (I - X * X^\top) * y = y - \sum_{j} (X_j * X_j^\top) * y
        orth_block = proj * orth_block;   // repeat orthogonalization (stabilize)

	// perform final stabilzed QR
        qr.compute(orth_block);
        orth_block = qr.householderQ() * matrix_t::Identity(rows, cols);
        return std::make_pair(orth_block, qr.matrixQR().triangularView<Eigen::Upper>().toDenseMatrix().topRows(cols));
    }

    double tol_ = 1e-5;
    int max_iter_ = 50;
    int seed_;

    matrix_t U_;
    vector_t Lambda_;
    int rank_;
};
  
}   // namespace fdapde

#endif // __RBKI_H__
