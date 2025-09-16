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

#ifndef __RANDOMLY_PERMUTED_CHOLESKY_H__
#define __RANDOMLY_PERMUTED_CHOLESKY_H__

namespace fdapde {

// computes a low rank approximation of an SPD matrix using the randomly pivoted cholesky, as detailed in "Y., Chen,
// E.N., Epperly, J.A., Tropp and R.J. Webber. (2023) Randomly pivoted Cholesky: Practical approximation of a kernel
// matrix with few entry evaluations", Alg 5, pag 12.
template <typename MatrixType>
    requires(internals::is_eigen_dense_xpr_v<MatrixType>)
class RpChol {
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using chol_t   = Eigen::LLT<matrix_t>;
   public:
  
    RpChol() noexcept = default;
    RpChol(const MatrixType& A, double tol, int block_sz, int max_iter, int seed = random_seed) noexcept :
        block_sz_(block_sz),
        max_iter_(max_iter),
        seed_(seed == random_seed ? std::random_device()() : seed) {
        compute(A, tol);
    }
    RpChol(int block_sz, int max_iter, int seed = random_seed) noexcept :
        block_sz_(block_sz), max_iter_(max_iter), seed_(seed == random_seed ? std::random_device()() : seed) { }
  
    void compute(const MatrixType& A, double tol) {
        fdapde_assert(tol < 1);
        int rows = A.rows();
        int cols = A.cols();

	// initialization
        int max_iter = std::min(max_iter_, int_ceil(cols, block_sz_));
        std::mt19937 rng(seed_);
        std::vector<int> col_idxs(cols);
        std::iota(col_idxs.begin(), col_idxs.end(), 0);
	vector_t diag_res = A.diagonal();
        L_ = matrix_t::Zero(rows, max_iter * block_sz_);
        double norm = A.norm();
	double err = norm;
	
        int i = 0;
        while (i < max_iter * block_sz_ && err > tol * norm) {
            // sample pivots with a probabilty proportional to diagonal elements of residuals
            std::discrete_distribution<int> distr(diag_res.begin(), diag_res.end());
            std::unordered_set<int> pivot_set(block_sz_);
            for (std::size_t j = 0; pivot_set.size() < block_sz_ && j < 2 * block_sz_; j++) {
                pivot_set.insert(distr(rng));
            }
            pivot_set_.merge(pivot_set);
            std::vector<int> pivot_vec(pivot_set.begin(), pivot_set.end());

	    for(int kk : pivot_vec) std::cout << kk << " ";
	    std::cout << std::endl;
	    
            // evaluate columns at pivot_set, remove overlap with previously choosen columns
            matrix_t G = A(Eigen::all, pivot_vec) - L_.leftCols(i) * L_(pivot_vec, Eigen::all).leftCols(i).transpose();
            // compute stabilized cholesky
            double shift = std::numeric_limits<double>::epsilon() * G(pivot_vec, Eigen::all).trace();
            chol_t chol(G(pivot_vec, Eigen::all) + shift * matrix_t::Identity(pivot_vec.size(), pivot_vec.size()));
            // update step
            L_.middleCols(i, pivot_vec.size()) = chol.matrixU().solve<Eigen::OnTheRight>(G);
            diag_res = (diag_res - L_.middleCols(i, pivot_vec.size()).rowwise().squaredNorm()).array().max(0);
            i += pivot_vec.size();
            err = (A - L_.leftCols(i) * L_.leftCols(i).transpose()).norm();
        }
        // store result
        L_ = L_.leftCols(i);
        return;
    }
    // observers
    const matrix_t& matrixL() const { return L_; }
    const std::unordered_set<int>& pivotSet() const { return pivot_set_; }
   private:
    matrix_t L_;                          // matrix L defining the Nystrom approximation of A \approx L_ * L_^\top
    std::unordered_set<int> pivot_set_;   // set of columns of A selected for the approximation

    int block_sz_;
    int max_iter_ = 50;
    int seed_;
};

}   // namespace fdapde

#endif // __RANDOMLY_PERMUTED_CHOLESKY_H__
