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

  template <typename SparseSolver = fdaPDE::SparseLU<SpMatrix<double>>,
	    typename DenseSolver  = Eigen::PartialPivLU<DMatrix<double>>>
  struct SMW{
    // constructor
    SMW() = default;
    
    // solves linear system (A + U*C^{-1}*V)x = b, assume to supply the already computed inversion of the dense matrix C.
    // This method must be executed after call to .compute()
    DMatrix<double> solve(const SparseSolver& invA, const DMatrix<double>& U, const DMatrix<double>& invC,
			  const DMatrix<double>& V, const DMatrix<double>& b){
      DMatrix<double> y = invA.solve(b); // y = A^{-1}b
      // Y = A^{-1}U. Heavy step of the method. SMW is more and more efficient as q gets smaller and smaller
      DMatrix<double> Y = invA.solve(U);
      // compute dense matrix G = C^{-1} + V*A^{-1}*U = C^{-1} + V*y
      DMatrix<double> G = invC + V*Y;
      DenseSolver invG; invG.compute(G); // factorize qxq dense matrix G
      DMatrix<double> t = invG.solve(V*y);
      // v = A^{-1}*U*t = A^{-1}*U*(C^{-1} + V*A^{-1}*U)^{-1}*V*A^{-1}*b by solving linear system A*v = U*t
      DMatrix<double> v = invA.solve(U*t);
      return y - v; // return system solution
    }
  };

}}}
#endif // _SMW_H__
