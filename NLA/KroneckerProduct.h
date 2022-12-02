#ifndef __KRONECKER_PRODUCT_H__
#define __KRONECKER_PRODUCT_H__

#include "../utils/Symbols.h"
#include <Eigen/Core>
#include <Eigen/src/Core/CoreEvaluators.h>
#include <Eigen/src/Core/CoreIterators.h>
#include <Eigen/src/Core/EigenBase.h>
#include <Eigen/src/Core/MatrixBase.h>
#include <type_traits>

// forward declaration
template <typename Lhs_, typename Rhs_> class KroneckerProduct;

// define template specialization to let KroneckerProduct integrate with Eigen
namespace Eigen {
namespace internal{

  // template specialization for KroneckerProduct traits (required by Eigen).
  template <typename Lhs, typename Rhs>
  struct traits<KroneckerProduct<Lhs, Rhs>> {
    // typedef required by eigen
    typedef typename std::decay<Lhs>::type LhsCleaned;
    typedef typename std::decay<Rhs>::type RhsCleaned;
    typedef traits<LhsCleaned> LhsTraits;
    typedef traits<RhsCleaned> RhsTraits;

    typedef Eigen::MatrixXpr XprKind; // expression type (matrix)
    // type of coefficients handled by this operator
    typedef typename ScalarBinaryOpTraits<
      typename traits<LhsCleaned>::Scalar,
      typename traits<RhsCleaned>::Scalar>::ReturnType Scalar;
    // storage informations
    typedef typename product_promote_storage_type<
      typename LhsTraits::StorageKind,
      typename RhsTraits::StorageKind,
      internal::product_type<Lhs,Rhs>::ret>::ret StorageKind;
    typedef typename promote_index_type<
      typename LhsTraits::StorageIndex,
      typename RhsTraits::StorageIndex>::type StorageIndex;

    enum { // definition of required compile time informations
      Flags = Eigen::ColMajor,
      RowsAtCompileTime = (Lhs::RowsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic)
         ? Dynamic : Lhs::RowsAtCompileTime*Rhs::RowsAtCompileTime,
      ColsAtCompileTime = (Lhs::ColsAtCompileTime == Dynamic || Rhs::ColsAtCompileTime == Dynamic)
         ? Dynamic : Lhs::ColsAtCompileTime*Rhs::ColsAtCompileTime,
      MaxRowsAtCompileTime = (Lhs::MaxRowsAtCompileTime == Dynamic || Rhs::MaxRowsAtCompileTime == Dynamic)
         ? Dynamic : Lhs::MaxRowsAtCompileTime*Rhs::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = (Lhs::MaxColsAtCompileTime == Dynamic || Rhs::MaxColsAtCompileTime == Dynamic)
         ? Dynamic : Lhs::MaxColsAtCompileTime*Rhs::MaxColsAtCompileTime
    };
  };
  
  // where the actual computation is defined
  template <typename Lhs_, typename Rhs_>
  struct evaluator<KroneckerProduct<Lhs_, Rhs_>> : evaluator_base<KroneckerProduct<Lhs_, Rhs_>> {
    // typedefs expected by eigen internals
    typedef KroneckerProduct<Lhs_, Rhs_> XprType; 
    // only one access is needed in the nested expression for each access to the evaluator
    typedef typename nested_eval<Lhs_, 1>::type LhsNested;
    typedef typename std::decay<LhsNested>::type LhsNestedCleaned;
    typedef typename nested_eval<Rhs_, 1>::type RhsNested; 
    typedef typename std::decay<RhsNested>::type RhsNestedCleaned;
    // type of coefficient returned by a single access to the evaluator
    typedef typename XprType::CoeffReturnType CoeffReturnType; 

    enum {
      CoeffReadCost = evaluator<LhsNestedCleaned>::CoeffReadCost + evaluator<RhsNestedCleaned>::CoeffReadCost,
      Flags = Eigen::ColMajor // only ColMajor storage orders accepted
    };
    
    evaluator<LhsNestedCleaned> lhs_; // left hand operand
    evaluator<RhsNestedCleaned> rhs_; // right hand operand
    const KroneckerProduct<Lhs_, Rhs_>& xpr_;
    // constructor
    evaluator(const KroneckerProduct<Lhs_, Rhs_>& xpr) : xpr_(xpr), lhs_(xpr.lhs_), rhs_(xpr.rhs_) {};
    // evaluate the (i,j)-th element of the kronecker product between the identity matrix and rhs_.
    CoeffReturnType coeff(Eigen::Index row, Eigen::Index col) const {
      return lhs_.coeff(row / xpr_.rhs_rows_, col / xpr_.rhs_cols_) * rhs_.coeff(row % xpr_.rhs_rows_, col % xpr_.rhs_cols_);
    }
  };
}}

// Eigen extension implementing the Kronecker tensor product between matrices.
// This class implements a node in an eigen expression-tree
template <typename Lhs_, typename Rhs_>
class KroneckerProduct : public Eigen::MatrixBase<KroneckerProduct<Lhs_, Rhs_>>{
public:
  // typedefs expected by eigen internals
  typedef Lhs_ Lhs;
  typedef Rhs_ Rhs;  
  typedef typename Eigen::internal::ref_selector<Lhs>::type LhsNested;
  typedef typename Eigen::internal::ref_selector<Rhs>::type RhsNested;
  // required to let KroneckerProduct integration with Eigen expression templates
  typedef typename Eigen::internal::ref_selector<KroneckerProduct>::type Nested;
  
  // expression operands
  LhsNested lhs_;
  RhsNested rhs_;
  Eigen::Index rhs_rows_, rhs_cols_;

  
  // constructor
  KroneckerProduct(const Lhs& lhs, const Rhs& rhs)
    : lhs_(lhs), rhs_(rhs), rhs_rows_(rhs_.rows()), rhs_cols_(rhs_.cols()){};
  // dimensions of the resulting kronecker product
  inline Eigen::Index rows() const { return lhs_.rows()*rhs_.rows(); }
  inline Eigen::Index cols() const { return lhs_.cols()*rhs_.cols(); }
};

// returns the kronecker product between lhs and rhs as an eigen expression
template <typename Lhs, typename Rhs>
KroneckerProduct<Lhs, Rhs> Kronecker(const Eigen::MatrixBase<Lhs> &lhs, const Eigen::MatrixBase<Rhs>& rhs) {
  return KroneckerProduct<Lhs, Rhs>(lhs.derived(), rhs.derived());
}

// forward declaration
template <typename Lhs_, typename Rhs_> class SparseKroneckerProduct;

// define template specialization to let KroneckerProduct integrate with Eigen
namespace Eigen {
namespace internal{

  // template specialization for KroneckerProduct traits (required by Eigen).
  template <typename Lhs, typename Rhs>
  struct traits<SparseKroneckerProduct<Lhs, Rhs>> : public traits<KroneckerProduct<Lhs, Rhs>>{};
  
  // where the actual computation is defined
  template <typename Lhs_, typename Rhs_>
  struct evaluator<SparseKroneckerProduct<Lhs_, Rhs_>> : evaluator_base<SparseKroneckerProduct<Lhs_, Rhs_>> {
    // typedefs expected by eigen internals
    typedef SparseKroneckerProduct<Lhs_, Rhs_> XprType; 
    // only one access is needed in the nested expression for each access to the evaluator
    typedef typename nested_eval<Lhs_, 1>::type LhsNested;
    typedef typename std::decay<LhsNested>::type LhsNestedCleaned;
    typedef typename nested_eval<Rhs_, 1>::type RhsNested; 
    typedef typename std::decay<RhsNested>::type RhsNestedCleaned;
    // type of coefficient returned by a single access to the evaluator
    typedef typename XprType::CoeffReturnType CoeffReturnType; 
    
    enum {
      CoeffReadCost = evaluator<LhsNestedCleaned>::CoeffReadCost + evaluator<RhsNestedCleaned>::CoeffReadCost,
      Flags = Eigen::ColMajor // only ColMajor storage orders accepted
    };
    
    evaluator<Lhs_> lhs_; // left hand operand
    evaluator<Rhs_> rhs_; // right hand operand
    const SparseKroneckerProduct<Lhs_, Rhs_>& xpr_;

    // sparse evaluators need to provide an InnerIterator.
    // Definition of InnerIterator providing the kronecker tensor product of the operands
    typedef typename evaluator<Lhs_>::InnerIterator LhsIterator;
    typedef typename evaluator<Rhs_>::InnerIterator RhsIterator;
    // iterator definition
    class InnerIterator { 
    public:
      // usefull typedefs
      typedef typename traits<XprType>::Scalar Scalar;
      typedef typename traits<XprType>::StorageIndex StorageIndex;
      // costructor
      InnerIterator(const evaluator<XprType>& m, Index outer)
	: lhs_it(m.lhs_, outer / m.xpr_.rhs_.outerSize()), rhs_it(m.rhs_, outer % m.xpr_.rhs_.outerSize()), m_(m),
	  m_index(lhs_it.index()*m.xpr_.rhs_.innerSize() - 1), outer_(outer) {
	// init iterator
	this->operator++();
      };

      InnerIterator& operator++() {
	if(rhs_it && lhs_it){
	  m_index++;
	  m_value = lhs_it.value() * rhs_it.value();
	  ++rhs_it;
	}else if(!rhs_it && ++lhs_it){
	  ++blk_;
	  rhs_it = RhsIterator(m_.rhs_, outer_ % m_.xpr_.rhs_.outerSize());
	  m_index = blk_*m_.xpr_.rhs_.innerSize();
	  m_value = lhs_it.value() * rhs_it.value();
	  ++rhs_it;	
	}else{
	  m_index = -1;
	}
	return *this;
      };

      operator bool() const { return m_index >= 0; }
      inline Index col() const { return col_; }
      inline Index row() const { return row_; }
      
      inline Index outer() const { return rhs_it.outer(); }
      inline Scalar value() const { return m_value; }
      inline StorageIndex index() const { return m_index; }
      
    protected:
      // reference to operands' iterator
      LhsIterator lhs_it;
      RhsIterator rhs_it;
      const evaluator<XprType>& m_;
      Scalar m_value; // the value pointed by the iterator
      
      StorageIndex m_index;
      Index row_, col_;
      Index blk_ = 0;
      Index outer_;
    };
 
    // constructor
    evaluator(const SparseKroneckerProduct<Lhs_, Rhs_>& xpr) : xpr_(xpr), lhs_(xpr.lhs_), rhs_(xpr.rhs_) {};
  };
}}

// Eigen extension implementing the Kronecker tensor product between matrices.
// This class implements a node in an eigen expression-tree
template <typename Lhs_, typename Rhs_>
class SparseKroneckerProduct : public Eigen::SparseMatrixBase<SparseKroneckerProduct<Lhs_, Rhs_>>{
public:
  // typedefs expected by eigen internals
  typedef Lhs_ Lhs;
  typedef Rhs_ Rhs;  
  typedef typename Eigen::internal::ref_selector<Lhs>::type LhsNested;
  typedef typename Eigen::internal::ref_selector<Rhs>::type RhsNested;
  // required to let KroneckerProduct integration with Eigen expression templates
  typedef typename Eigen::internal::ref_selector<SparseKroneckerProduct>::type Nested;
  
  // expression operands
  LhsNested lhs_;
  RhsNested rhs_;  
  // constructor
  SparseKroneckerProduct(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) {};
  // dimensions of the resulting kronecker product
  inline Eigen::Index rows() const { return lhs_.rows()*rhs_.rows(); }
  inline Eigen::Index cols() const { return lhs_.cols()*rhs_.cols(); }
};

// sparse operands
template <typename Lhs, typename Rhs>
SparseKroneckerProduct<Lhs, Rhs> Kronecker(const Eigen::EigenBase<Lhs> &lhs, const Eigen::EigenBase<Rhs>& rhs) {
  return SparseKroneckerProduct<Lhs, Rhs>(lhs.derived(), rhs.derived());
}

#endif // __KRONECKER_PRODUCT_H__
