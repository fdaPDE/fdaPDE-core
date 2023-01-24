#ifndef __SMW_H__
#define __SMW_H__

#include "../utils/Symbols.h"
#include <Eigen/SparseLU>
#include <Eigen/LU>

namespace fdaPDE {
namespace core{
namespace NLA{

  // A general implementation of a system solver using the Sherman–Morrison–Woodbury formula for matrix inversion

  // Consider a linear system Mx = b. The direct solution of this system would require the inversion of M, i.e. x = M^{-1}*b.
  // If we are able to decompose matrix M as (A + UCV) we can take advantage of the SMW formula to compute
  //
  //        (A + UCV)^{-1} = A^{-1} - A^{-1}*U*(C^{-1} + V*A^{-1}*U)^{-1}*V*A^{-1}
  //
  // In case A is a sparse matrix and C a small dense matrix, computing M^{-1} using the above decomposition is much more efficient
  // than computing M^{-1} directly

  template <typename SparseSolver = Eigen::SparseLU<SpMatrix<double>, Eigen::COLAMDOrdering<int>>,
	    typename DenseSolver  = Eigen::PartialPivLU<DMatrix<double>>>
  class SMW{
  private:
    SparseSolver sparseSolver_; // store factorization of 2Nx2N sparse matrix A
    DenseSolver  denseSolver_;  // store factorization of dense matrix G = C^{-1} + V*A^{-1}*U
  public:
    SMW() = default;
    
    void compute(const SpMatrix<double>& A){
      // compute sparse factorization of matrix A and store it for fast reuse
      sparseSolver_.analyzePattern(A); // Compute the ordering permutation vector from the structural pattern of A
      sparseSolver_.factorize(A);      // compute LU factorization of matrix A
    }
    
    // solves linear system (A + U*C^{-1}*V)x = b, observe that we assume to supply the already computed inversion of the dense matrix C.
    // Indeed in some cases the inverse of C can be directly computed from known quantites without performing any inversion. In the case
    // this is not possible, you can find the inverse of C and then call this .solve() method.
    // A 2Nx2N sparse matrix, U 2Nxq sparse matrix, invC qxq dense matrix, V qx2N sparse matrix, b 2Nx1 vector.
    // In a regression model q is the number of covariates. This method must be executed after call to .compute()
    DMatrix<double> solve(const DMatrix<double>& U, const DMatrix<double>& invC, const DMatrix<double>& V, const DMatrix<double>& b){
      // split the solution of the linear system (A + U*C^{-1}*V)x = b in the solution of 3 computationally simpler linear systems

      // compute y = A^{-1}b
      DMatrix<double> y = sparseSolver_.solve(b);
      // compute Y = A^{-1}U. Heavy step of the method. SMW is more and more efficient as q gets smaller and smaller
      DMatrix<double> Y = sparseSolver_.solve(U);
      // compute dense matrix G = C^{-1} + V*A^{-1}*U = C^{-1} + V*y
      DMatrix<double> G = invC + V*Y;
      denseSolver_.compute(G); // factorize G
      DMatrix<double> t = denseSolver_.solve(V*y); // solve dense qxq linear system
      // compute v = A^{-1}*U*t = A^{-1}*U*(C^{-1} + V*A^{-1}*U)^{-1}*V*A^{-1}*b by solving linear system A*v = U*t
      DMatrix<double> v = sparseSolver_.solve(U*t);
      // return system solution
      return y - v;
    }

    // getters
    const SparseSolver& getSparseSolver() const { return sparseSolver_; }
    const DenseSolver&  getDenseSolver()  const { return denseSolver_;  }
  
  };

}}}
#endif // _SMW_H__
