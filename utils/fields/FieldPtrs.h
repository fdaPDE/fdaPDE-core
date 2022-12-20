#ifndef __FIELD_PTRS_H__
#define __FIELD_PTRS_H__

#include "expressions/ScalarExpressions.h"
using fdaPDE::core::ScalarExpr;
#include "expressions/VectorExpressions.h"
using fdaPDE::core::VectorExpr;
#include "expressions/MatrixExpressions.h"
using fdaPDE::core::MatrixExpr;

namespace fdaPDE{
namespace core{
 
  // basic pointer type for scalar expressions
  template <typename E> class ScalarPtr
    : public ScalarExpr<ScalarPtr<E>> {
    static_assert(std::is_base_of<ScalarBase, E>::value);
  private:
    typename std::remove_reference<E>::type* ptr_;
  public:
    // constructor
    ScalarPtr(E* ptr) : ptr_(ptr) {};
    // delegate to pointed memory location
    template <int N>
    double operator()(const SVector<N>& p) const{
      return ptr_->operator()(p);
    }
    // delegate to pointed memory location
    template <typename T> void eval_parameters(T i) {
      ptr_->eval_parameters(i);
      return;
    }
    // access to pointed element
    E* operator->() { return ptr_; }
    typedef E PtrType; // expose wrapped type
  };

  // basic pointer type for vectorial expressions
  template <typename E> class VectorPtr
    : public VectorExpr<E::base, E::rows, VectorPtr<E>> {
    static_assert(std::is_base_of<VectorBase, E>::value);
  private:
    typename std::remove_reference<E>::type* ptr_;
  public:
    VectorPtr(E* ptr) : ptr_(ptr) {};
    // delegate to pointed memory location
    auto operator[](std::size_t i) const{
      return ptr_->operator[](i);
    }
    // delegate to pointed memory location
    template <typename T> void eval_parameters(T i) {
      ptr_->eval_parameters(i);
      return;
    }
    // access to pointed element
    E* operator->() { return ptr_; }
    typedef E PtrType; // expose wrapped type
  };

  // basic pointer type for matrix expressions
  template <typename E> class MatrixPtr
    : public MatrixExpr<E::base, E::rows, E::cols, MatrixPtr<E>> {
    static_assert(std::is_base_of<MatrixBase, E>::value);
  private:
    typename std::remove_reference<E>::type* ptr_;
  public:
    MatrixPtr(E* ptr) : ptr_(ptr) {};
    // delegate to pointed memory location
    auto coeff(std::size_t i, std::size_t j) const {
      return ptr_->coeff(i,j);
    }
    // delegate to pointed memory location
    template <typename T>
    void eval_parameters(T i) { ptr_->eval_parameters(i); }
    // access to pointed element
    E* operator->() { return ptr_; }
    typedef E PtrType; // expose wrapped type
  };
  
}}

#endif // __FIELD_PTRS_H__
