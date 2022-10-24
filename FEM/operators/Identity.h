#ifndef __IDENTITY_H__
#define __IDENTITY_H__

#include <type_traits>
#include "../../utils/fields/ScalarField.h"
using fdaPDE::core::ScalarField;
#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;
#include "BilinearFormExpressions.h"
using fdaPDE::core::FEM::BilinearFormExpr;

namespace fdaPDE{
namespace core{
namespace FEM{

  // class representing the identity operator (reaction term)
  template <typename T = NullaryOperator>
  class Identity : public BilinearFormExpr<Identity<T>>{
    // perform compile-time sanity checks
    static_assert(std::is_same<NullaryOperator, T>::value || // support for dot(b, Gradient())
		  std::is_base_of<ScalarBase, T>::value || // space-varying case
		  std::is_floating_point<T>::value); // constant coefficient case
  private:
    T c_; // reaction term
  public:
    // constructors
    Identity() = default;
    Identity(const T& c) : c_(c) {};

    std::tuple<Identity<T>> getTypeList() const { return std::make_tuple(*this); }

    // approximates the contribution to the (i,j)-th element of the discretization matrix given by the transport term:
    // \int_e phi_i * phi_j
    // basis: any type compliant with a functional basis behaviour. See LagrangianBasis.h for an example
    //        NOTE: we assume "basis" to provide functions already defined on the reference element
    // e: the element on which we are integrating
    // i,j: indexes of the discretization matrix element we are computing
    template <unsigned int M, unsigned int N, unsigned int R, typename B>
    ScalarField<M> integrate(const B& basis, const Element<M, N, R>& e, int i , int j) const{
      auto phi_i = basis[i];  
      auto phi_j = basis[j];
      // approximation of the (i,j)-th element of identity operator
      if constexpr(std::is_same<NullaryOperator, T>::value){
	// fallback to c_ = 1
	return phi_i * phi_j;
      }else{
	if constexpr(std::is_base_of<ScalarBase, T>::value){
	  std::function<double(SVector<M>)> c = [this, e](const SVector<M>& p) -> double {
	    // when the bilinear form is integrated it gets quadrature nodes defined over the reference element.
	    // we need to map the quadrature point on the physical element e to get a correct evaluation of the non-constant field c_
	    return c_(e.barycentricMatrix()*p + e.coords()[0]);
	  };
	  return c *(phi_i*phi_j);
	}else{
	  return c_*(phi_i*phi_j);
	}
      }
    }
  };
  // template argument deduction guide
  template <typename T> Identity(const T&) -> Identity<T>;
  
}}}
#endif // __IDENTITY_H__
