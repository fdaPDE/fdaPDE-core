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

#ifndef __RSI_H__
#define __RSI_H__

namespace fdapde {

// randomized SVD decomposition of a matrix, as described in
// J. A. Tropp and R. J. Webber. (2023) Randomized algorithms for low-rank matrix approximation: Design, analysis, and
// applications, Alg 5.2, pag 16.
template <typename MatrixType>
    requires(internals::is_eigen_dense_xpr_v<MatrixType>)
class RSI {
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using qr_t     = Eigen::HouseholderQR<matrix_t>;
    using svd_t    = Eigen::JacobiSVD<matrix_t>;
   public:

    RSI() noexcept =default;
    RSI(const MatrixType& m, int rank) noexcept : tol_(1e-5), max_iter_(50), seed_(std::random_device()()) {
        compute(m, rank);
    }
    RSI(const MatrixType& m, int rank, double tol, int max_iter, int seed = random_seed) noexcept :
        tol_(tol), max_iter_(max_iter), seed_(seed == random_seed ? std::random_device()() : seed) {
        compute(m, rank);
    }
    RSI(double tol, int max_iter, int seed = random_seed) noexcept :
        tol_(tol), max_iter_(max_iter), seed_(seed == random_seed ? std::random_device()() : seed) { }

    // computes the decomposition
    void compute(const MatrixType& A, int rank) {
        // use as block size the one suggested in "N. Halko, P.G. Martinsson, and J. A. Tropp. (2010) Finding structure
        // with randomness: Probabilistic algorithms for constructing approximate matrix decompositions."
        compute(A, rank, fdapde::min(2 * rank, fdapde::min(A.rows(), A.cols())));
    }
    void compute(const MatrixType& A, int rank, int block_sz) {
        rank_ = rank;
        int rows = A.rows();
        int cols = A.cols();
        // approximate range of A
        matrix_t Omega = internals::gaussian_matrix(cols, block_sz, 1.0, seed_);
        qr_t qr(A * Omega);
        matrix_t Q = qr.householderQ() * matrix_t::Identity(rows, block_sz);
        matrix_t B = A.transpose() * Q;
        // subspace iteration loop initialization
	int iter = 0;
        svd_t svd(B.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        matrix_t Ar = A * svd.matrixV().leftCols(rank);
        matrix_t E = Ar - Q * svd.matrixU().leftCols(rank) * svd.singularValues().head(rank).asDiagonal();
        double res_err = E.colwise().template lpNorm<2>().maxCoeff();
        // computes (A * A^\top)^{q} * A * \Omega
        for (; res_err > tol_ && iter < max_iter_; iter++) {
            qr.compute(B);
            Q = qr.householderQ() * matrix_t::Identity(cols, block_sz);
            B = A * Q;
            qr.compute(B);
            Q = qr.householderQ() * matrix_t::Identity(rows, block_sz);
            B = A.transpose() * Q;
            // residual error update
            svd.compute(B.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            E = Ar - Q * svd.matrixU().leftCols(rank) * svd.singularValues().head(rank).asDiagonal();
            res_err = E.colwise().template lpNorm<2>().maxCoeff();
        }
        // store result
        U_ = Q * svd.matrixU().leftCols(rank);
        V_ = svd.matrixV().leftCols(rank);
        Sigma_ = svd.singularValues().head(rank);
        return;
    }
    // observers
    const matrix_t& matrixU() const { return U_; }
    const matrix_t& matrixV() const { return V_; }
    const vector_t& singularValues() const { return Sigma_; }
    int rank() const { return rank_; }
   private:
    double tol_ = 1e-5;
    int max_iter_ = 50;
    int seed_;

    matrix_t U_, V_;
    vector_t Sigma_;
    int rank_;
};

// randomized eigenvalue decomposition of a (SPD) matrix, as described in
// J. A. Tropp and R. J. Webber. (2023) Randomized algorithms for low-rank matrix approximation: Design, analysis, and
// applications, Alg 5.7, pag 21.
template <typename MatrixType>
    requires(internals::is_eigen_dense_xpr_v<MatrixType>)
class NysRSI {
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using qr_t     = Eigen::HouseholderQR<matrix_t>;
    using chol_t   = Eigen::LLT<matrix_t>;
    using svd_t    = Eigen::JacobiSVD<matrix_t>;
   public:
    NysRSI() noexcept =default;
    NysRSI(const MatrixType& m, int rank) noexcept : tol_(1e-5), max_iter_(50), seed_(std::random_device()()) {
        compute(m, rank);
    }
    NysRSI(const MatrixType& m, int rank, double tol, int max_iter, int seed = random_seed) noexcept :
        tol_(tol), max_iter_(max_iter), seed_(seed == random_seed ? std::random_device()() : seed) {
        compute(m, rank);
    }
    NysRSI(double tol, int max_iter, int seed = random_seed) noexcept :
        tol_(tol), max_iter_(max_iter), seed_(seed == random_seed ? std::random_device()() : seed) { }

    // computes the decomposition
    void compute(const MatrixType& A, int rank) { compute(A, rank, fdapde::min(2 * rank, A.rows())); }
    void compute(const MatrixType& A, int rank, int block_sz) {
        rank_ = rank;
        int rows = A.rows();
        double shift = A.diagonal().sum() * std::numeric_limits<double>::epsilon();   // epsilon_shift
	
        // subspace iteration loop initialization
        matrix_t Y = internals::gaussian_matrix(rows, block_sz, 1.0, seed_);
        svd_t svd;
        int iter = 0;
        double res_err = std::numeric_limits<double>::max();
        for(; res_err > tol_ && iter < max_iter_; ++iter) {
            // A's range approximation
            qr_t qr(Y);
            matrix_t X = qr.householderQ() * matrix_t::Identity(rows, block_sz);
            Y = A * X;
            // Nystrom factor computation
            Y = Y + shift * matrix_t::Identity(Y.rows(), Y.cols());
            chol_t chol(X.transpose() * Y);
            matrix_t F = chol.matrixU().solve<Eigen::OnTheRight>(Y);   // F = Y * C^{-1}, with X^\top * Y = C^\top * C
            // residual error update
            svd.compute(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
            matrix_t E = A * svd.matrixU().leftCols(rank) -
                svd.matrixU().leftCols(rank) *
                  (svd.singularValues().head(rank).array().pow(2) - shift).matrix().asDiagonal();
            res_err = std::sqrt(2) * E.colwise().template lpNorm<2>().maxCoeff();
        }
        // store result
        U_ = svd.matrixU().leftCols(rank);
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
    double tol_ = 1e-5;
    int max_iter_ = 50;
    int seed_;

    matrix_t U_;
    vector_t Lambda_;
    int rank_;
};

}   // namespace fdapde

#endif // __RSI_H__
