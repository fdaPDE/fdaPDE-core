#ifndef __KRONECKER_PRODUCT_H__
#define __KRONECKER_PRODUCT_H__

#include "../utils/Symbols.h"
#include <Eigen/Core>
#include <type_traits>

// Implementation of the Kronecker tensor product as an Eigen expression type. This allows to write expressions
// involving the Kronecker product of some matrices which are lazy evaluated as part of the Eigen expression
// template mechanism.

// forward declarations
namespace fdaPDE {
namespace core {
namespace NLA {

  // definition of the dense-dense Kronecker product as a node in an eigen expression-tree
  template <typename Lhs_, typename Rhs_>
  struct KroneckerProduct : public Eigen::MatrixBase<KroneckerProduct<Lhs_, Rhs_>> {
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
    // constructor
    KroneckerProduct(const Lhs& lhs, const Rhs& rhs)
      : lhs_(lhs), rhs_(rhs) {};
    // dimensions of the resulting kronecker product
    inline Eigen::Index rows() const { return lhs_.rows()*rhs_.rows(); }
    inline Eigen::Index cols() const { return lhs_.cols()*rhs_.cols(); }
  };

  // definition of the sparse-sparse Kronecker product as a node in an eigen expression-tree
  template <typename Lhs_, typename Rhs_>
  struct SparseKroneckerProduct : public Eigen::SparseMatrixBase<SparseKroneckerProduct<Lhs_, Rhs_>>{
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

  // returns the kronecker product between lhs and rhs as an eigen expression (dense version)
  template <typename Lhs, typename Rhs>
  KroneckerProduct<Lhs, Rhs> Kronecker
  (const Eigen::MatrixBase<Lhs> &lhs, const Eigen::MatrixBase<Rhs>& rhs) {
    return KroneckerProduct<Lhs, Rhs>(lhs.derived(), rhs.derived());
  }

  // returns the kronecker product between lhs and rhs as an eigen expression (sparse version)
  template <typename Lhs, typename Rhs>
  SparseKroneckerProduct<Lhs, Rhs> Kronecker
  (const Eigen::SparseMatrixBase<Lhs> &lhs, const Eigen::SparseMatrixBase<Rhs>& rhs) {
    return SparseKroneckerProduct<Lhs, Rhs>(lhs.derived(), rhs.derived());
  }

}}}
      
// definition of proper symbols in Eigen::internal namespace
namespace Eigen {
namespace internal{
  // import symbols from fdaPDE namespace
  using fdaPDE::core::NLA::KroneckerProduct;
  using fdaPDE::core::NLA::SparseKroneckerProduct;
  
  // template specialization for KroneckerProduct traits (required by Eigen).
  template <typename Lhs_, typename Rhs_>
  struct traits<KroneckerProduct<Lhs_, Rhs_>> {
    typedef Lhs_ Lhs; typedef Rhs_ Rhs; // export operands type
    // typedef required by eigen
    typedef typename std::decay<Lhs>::type LhsCleaned;
    typedef typename std::decay<Rhs>::type RhsCleaned;
    typedef traits<LhsCleaned> LhsTraits;
    typedef traits<RhsCleaned> RhsTraits;

    typedef Eigen::MatrixXpr XprKind; // expression type (matrix-expression)
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

  // trait specialization for the sparse-sparse version
  template <typename Lhs, typename Rhs>
  struct traits<SparseKroneckerProduct<Lhs, Rhs>> : public traits<KroneckerProduct<Lhs, Rhs>>{};

  // Eigen requires a specialization of the evaluator template for each expression type. An evaluator is
  // responsible for the computation of the entries of the result matrix. For dense operations we need to provide
  // an implementation of the coeff(Index, Index) method.
  template <typename Lhs_, typename Rhs_>
  class evaluator<KroneckerProduct<Lhs_, Rhs_>>
    : public evaluator_base<KroneckerProduct<Lhs_, Rhs_>> {
  private:
    const KroneckerProduct<Lhs_, Rhs_>& xpr_;
  public:
    // typedefs expected by eigen internals
    typedef KroneckerProduct<Lhs_, Rhs_> XprType; 
    // only one access to each nested evaluator is required (meaning of 1 in nested_eval<> below) to evaluate this evaluator
    typedef typename nested_eval<Lhs_, 1>::type LhsNested;
    typedef typename std::decay<LhsNested>::type LhsNestedCleaned;
    typedef typename nested_eval<Rhs_, 1>::type RhsNested; 
    typedef typename std::decay<RhsNested>::type RhsNestedCleaned;
    // type of coefficient returned by a single access to the evaluator
    typedef typename XprType::CoeffReturnType CoeffReturnType; 

    enum { // required compile time constants
      CoeffReadCost = evaluator<LhsNestedCleaned>::CoeffReadCost + evaluator<RhsNestedCleaned>::CoeffReadCost,
      Flags = Eigen::ColMajor // only ColMajor storage orders accepted
    };
    // Kronecker product operands
    evaluator<LhsNestedCleaned> lhs_;
    evaluator<RhsNestedCleaned> rhs_;
    
    // constructor
    evaluator(const KroneckerProduct<Lhs_, Rhs_>& xpr) : xpr_(xpr), lhs_(xpr.lhs_), rhs_(xpr.rhs_) {};
    // evaluate the (i,j)-th element of the kronecker product between lhs and rhs
    CoeffReturnType coeff(Eigen::Index row, Eigen::Index col) const {
      return lhs_.coeff(row / xpr_.rhs_.rows(), col / xpr_.rhs_.cols()) *
	     rhs_.coeff(row % xpr_.rhs_.rows(), col % xpr_.rhs_.cols());
    }
  };

  // an evaluator for the Sparse case requires to define an InnerIterator type allowing to iterate over the inner dimension
  // of the result matrix. For matrices in column major order an inner iterator must be able to iterate over the rows of the
  // result once fixed a column (outer dimension).
  template <typename Lhs_, typename Rhs_>
  class evaluator<SparseKroneckerProduct<Lhs_, Rhs_>>
    : public evaluator_base<SparseKroneckerProduct<Lhs_, Rhs_>> {
  private:
    const SparseKroneckerProduct<Lhs_, Rhs_>& xpr_;
  public:    
    // typedefs expected by eigen internals
    typedef SparseKroneckerProduct<Lhs_, Rhs_> XprType; 
    typedef typename evaluator<Lhs_>::InnerIterator LhsIterator;
    typedef typename evaluator<Rhs_>::InnerIterator RhsIterator;
    
    enum { // required compile time constants
      CoeffReadCost = evaluator<Lhs_>::CoeffReadCost + evaluator<Rhs_>::CoeffReadCost,
      Flags = Eigen::ColMajor // only ColMajor storage orders accepted
    };
    // Kronecker product operands
    evaluator<Lhs_> lhs_;
    evaluator<Rhs_> rhs_;

    // Definition of InnerIterator providing the kronecker tensor product of the operands.
    class InnerIterator { 
    public:
      // usefull typedefs
      typedef typename traits<XprType>::Scalar Scalar;
      typedef typename traits<XprType>::StorageIndex StorageIndex;
      // costructor (outer is the index of the column over which we are iterating).
      InnerIterator(const evaluator<XprType>& m, Index outer)
	: lhs_it(m.lhs_, outer / m.xpr_.rhs_.outerSize()),
	  rhs_it(m.rhs_, outer % m.xpr_.rhs_.outerSize()), 
	  m_index(lhs_it.index()*m.xpr_.rhs_.innerSize() - 1),
	  outer_(outer), m_(m) {
	// init iterator
	this->operator++();
      };

      InnerIterator& operator++() {
	if(rhs_it && lhs_it){ 
	  m_index = lhs_it.index()*m_.xpr_.rhs_.innerSize() + rhs_it.index();;
	  // (i,j)-th kronecker product value
	  m_value = lhs_it.value() * rhs_it.value();
	  ++rhs_it;
	}else if(!rhs_it && ++lhs_it){ // start new block a_{ij}*B[,j]
	  rhs_it = RhsIterator(m_.rhs_, outer_ % m_.xpr_.rhs_.outerSize());
	  m_index = lhs_it.index()*m_.xpr_.rhs_.innerSize() + rhs_it.index();
	  // (i,j)-th kronecker product value
	  m_value = lhs_it.value() * rhs_it.value();
	  ++rhs_it;	
	}else{
	  // end of the iterator
	  m_index = -1;
	}
	return *this;
      };
      // access methods
      inline Scalar value() const { return m_value; } // value pointed by the iterator      
      inline Index col() const { return outer_; } // current column (assume ColMajor order)
      inline Index row() const { return index(); } // current row (assume ColMajor order)
      inline Index outer() const { return outer_; } // outer index
      inline StorageIndex index() const { return m_index; } // inner index
      operator bool() const { return m_index >= 0; } // false when the iterator is over

    protected:
      // reference to operands' iterators
      LhsIterator lhs_it;
      RhsIterator rhs_it;
      const evaluator<XprType>& m_;
      Scalar m_value; // the value pointed by the iterator
      StorageIndex m_index; // current inner index      
      Index outer_; // outer index as received from the constructor
    }; 
    // constructor
    evaluator(const SparseKroneckerProduct<Lhs_, Rhs_>& xpr)
      : xpr_(xpr), lhs_(xpr.lhs_), rhs_(xpr.rhs_) {};
  };
  
}}
  
#endif // __KRONECKER_PRODUCT_H__
