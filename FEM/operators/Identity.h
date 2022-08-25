#ifndef __IDENTITY_H__
#define __IDENTITY_H__

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
  template <unsigned int L = 0>
  struct Identity : public BilinearFormExpr<Identity<L>>{
    // constructors
    Identity() = default;

    std::tuple<Identity> getTypeList() const { return std::make_tuple(*this); }

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
      return phi_i*phi_j;
    }
  };

}}}
#endif // __IDENTITY_H__
