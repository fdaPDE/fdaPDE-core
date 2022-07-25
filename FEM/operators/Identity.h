#ifndef __IDENTITY_H__
#define __IDENTITY_H__

#include "../../utils/fields/ScalarField.h"
using fdaPDE::core::ScalarField;
#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;

#include "BilinearFormExpressions.h"

// class representing the identity operator (reaction term)
template <unsigned int L = 0>
struct Identity : public BilinearFormExpr<Identity<L>>{

  std::tuple<Identity> getTypeList() const { return std::make_tuple(*this); }
  
  // provide the discretization for the identity operator. In particular this method implements the quadrature rule
  // for approximating the (i,j)-th element of the stiffness matrix \int_e [phi_i * phi_j]
  // integrate() will be called by Integrator as a result of the expression template expansion of the problem's bilinear form

  // basis: any type compliant with a functional basis behaviour. See LagrangianBasis for an example
  // e: the element where we are integrating
  // i,j: indexes of the stiffness matrix's element we are computing
  // quadrature_point: the point where to evaluate the integrand
  template <unsigned int N, int M, unsigned int ORDER, typename B>
  double integrate(const B& basis, const Element<ORDER, N>& e, int i , int j, const SVector<M>& quadrature_point) const{
    // NOTE: we assume "basis" to provide functions already defined on the reference element
    auto phi_i = basis[i];  
    auto phi_j = basis[j];
    
    // for the reaction term: phi_i * phi_j
    return (phi_i*phi_j)(quadrature_point);
  }
};

#endif // __IDENTITY_H__
