#ifndef __KRONECKER_PRODUCT_H__
#define __KRONECKER_PRODUCT_H__

#include "../utils/Symbols.h"

namespace fdaPDE {
namespace core {
namespace NLA {

  // a functor implementing the kronecker product between two sparse matrices
  template <typename T1, typename T2>
  class KroneckerProduct {
  private:
    // references to operands
    const T1& op1_;
    const T2& op2_;
  public:
    // constructor
    KroneckerProduct(const T1& op1, const T2& op2) : op1_(op1), op2_(op2) {};
    
    // evaluate kronecker product of operands
    SpMatrix<double> eval() const {
      // extract informations from operands
      Eigen::Index op1_rows = op1_.rows(),     op2_rows = op2_.rows();
      Eigen::Index op1_cols = op1_.cols(),     op2_cols = op2_.cols();
      Eigen::Index op1_nnz  = op1_.nonZeros(), op2_nnz  = op2_.nonZeros();
      // prepare room for destination
      SpMatrix<double> result;
      result.resize(op1_rows*op2_rows, op1_cols+op2_cols);
      // the number of non-zero elements of the tensor product of op1 and op2 is known in advantage
      result.reserve(op1_nnz*op2_nnz); 

      // fill result matrix row by row (assuming operands in ColMajor order)
      for (Eigen::Index it1_outer = 0; it1_outer < op1_.outerSize(); ++it1_outer){
	for (Eigen::Index it2_outer = 0; it2_outer < op2_.outerSize(); ++it2_outer){
	  for (typename T1::InnerIterator it1_inner(op1_, it1_outer); it1_inner; ++it1_inner){
	    // multiply op1_{ij} for each element in the i-th row of op2
	    for (typename T2::InnerIterator it2_inner(op2_, it2_outer); it2_inner; ++it2_inner){
	      // compute destination index
	      Eigen::Index i = it1_inner.row() * op2_rows + it2_inner.row();
	      Eigen::Index j = it1_inner.col() * op2_cols + it2_inner.col();
	      // perform insertion
	      result.insert(i,j) = it1_inner.value() * it2_inner.value();
	    }
	  }
	}
      }
      // should perform NRVO
      return result;
    }
    
  };

}}}

#endif // __KRONECKER_PRODUCT_H__
