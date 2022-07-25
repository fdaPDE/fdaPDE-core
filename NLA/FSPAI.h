#ifndef __FSPAI_H__
#define __FSPAI_H__

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <vector>

#include "../utils/Symbols.h"

namespace fdaPDE{
namespace core{
namespace NLA{
    
  // An implementation of the Factorized Sparse Approximate Inverse algorithm with sparsity pattern update.
  // FSPAI assumes that the square, n x n, sparse matrix A of which we want to compute the inverse is SPD, in this sense there exists a lower
  // triangular matrix L_A such that A = L_A.transpose()*L_A. FSPAI finds an approximate inverse for the Cholesky factor L_A of matrix A
  // while keeping L_A sparse (in general indeed the inverse of a sparse matrix is not generally sparse, i.e. it could be dense)

  // This FSPAI implementation is based on the minimization of the K-condition number of matrix A. A must be SPD
  class FSPAI{
  private:
    // a sparsity pattern is a set of indexes corresponding to non-zero entries of the matrix
    typedef std::unordered_map<Eigen::Index, std::unordered_set<Eigen::Index>> sparsity_pattern;
    typedef std::unordered_set<std::size_t> column_sparsity_pattern;
    typedef Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> SPDsolver;

    const Eigen::SparseMatrix<double>& A_;    // const reference to target matrix
    Eigen::SparseMatrix<double> L_;           // the sparse approximate inverse of the Cholesky factor of A_
  
    // internal status data members
    Eigen::Index n_;                              // dimension of square sparse matrix A_
    std::vector<column_sparsity_pattern> J_;      // sparsity pattern of approximate inverse
    Eigen::Matrix<double, Eigen::Dynamic, 1> Lk_; // the k-th column of the approximate inverse of the cholesky factor L_
    SPDsolver choleskySolver_{}; // solver internally used for the solution of linear system A(tildeJk, tildeJk)*yk = Ak(tildeJk)

    // data structures used for efficiency reasons
    sparsity_pattern sparsityPattern_{};               // the sparsity pattern of matrix A_ stored in RowMajor mode
    std::unordered_set<Eigen::Index> candidateSet_{};  // set of candidate indexes to enter in the sparsity pattern of column Lk_
    std::unordered_set<Eigen::Index> deltaPattern_{};  // set of new indexes entering the sparsity pattern of column Lk_ wrt previous iteration
    std::unordered_map<Eigen::Index, double> hatJk_{}; // stores the tau_jk values of indexes eligible to enter the sparsity pattern of column Lk
  
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Ak_; // value of A(tildeJk, tildeJk). Stored here to avoid expensive copies
    Eigen::Matrix<double, Eigen::Dynamic, 1> bk_;              // value of Ak(tildeJk). Stored here to avoid expensive copies
  
    // build system matrix A(p1, p2) given sparsity patterns p1 and p2. The result is a |p1| x |p2| dense matrix
    void extractSystem(const column_sparsity_pattern& p1,
		       const column_sparsity_pattern& p2,
		       const Eigen::Index& k);
    // update approximate inverse of column k
    void updateApproximateInverse(const Eigen::Index& k, const DVector<double>& bk, const DVector<double>& yk,
				  const column_sparsity_pattern& tildeJk);
    // select candidate indexes for sparsity pattern update for column k (this reflects in a modification to the hatJk_ structure)
    void selectCandidates(const Eigen::Index& k);
  
  public:
    // constructor
    FSPAI(const Eigen::SparseMatrix<double>& A);
    // returns the approximate inverse of the Cholesky factor of matrix A_
    const Eigen::SparseMatrix<double>& getL() const { return L_; }
    // returns the approximate inverse of A_
    Eigen::SparseMatrix<double> getInverse()  const { return L_*L_.transpose(); }
  
    // compute the Factorize Sparse Approximate Inverse of A using a K-condition number minimization method
    // alpha:   number of sparsity pattern updates to compute for each column k of A_
    // beta:    number of indexes to augment the sparsity pattern of Lk_ per update step
    // epsilon: do not consider an entry of A_ as valid if it causes a reduction to its K-condition number lower than epsilon
    void compute(unsigned alpha, unsigned beta, double epsilon);
  };
  
}}}
  
#endif // __FSPAI_H__
