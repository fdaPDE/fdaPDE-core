#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#include "../../utils/Symbols.h"
#include "../../utils/fields/VectorField.h"
using fdaPDE::core::VectorField;
#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;

#include "BilinearFormExpressions.h"

// class representing the gradient operator (transport term)
// requires C++17 standard to allow for automatic class template argument deduction

template <unsigned int L = 0>
class Gradient : public BilinearFormExpr<Gradient<L>>{
private:
  SVector<L> b_;

public:
  Gradient() = default;
  Gradient(const SVector<L>& b) : b_(b) {}

  std::tuple<Gradient<L>> getTypeList() const { return std::make_tuple(*this); }
  
  // provide the discretization for the gradient operator. In particular this method implements a custom quadrature rule
  // for approximating the (i,j)-th element of the stiffness matrix \int_e phi_i * b.dot(\Nabla phi_j)
  // integrate() will be called by Integrator as a result of the expression template expansion of the problem's bilinear form

  // basis: any type compliant with a functional basis behaviour. See LagrangianBasis.h for an example
  // e: the element where we are integrating
  // i,j: indexes of the stiffness matrix element we are computing
  // quadrature_point: the point where to evaluate the integrand
  template <unsigned int M, unsigned int N, unsigned int R, typename Q, typename B>
  double integrate(const B& basis, const Element<M, N, R>& e, int i , int j, const Q& quadrature_point) const{
    // express gradient of basis function over e in terms of gradients of basis functions over reference element.
    // This entails to compute (J^{-1})^T * \Nabla phi_i. In the following we assume basis[i] = phi_i
    Eigen::Matrix<double, N, M> invJ = e.invBarycentricMatrix().transpose();
    // Given \Nabla phi_i premultiply it by (J^{-1})^T = invJ.
    // NOTE: we assume "basis" to provide functions already defined on the reference
    VectorField<M, N> NablaPhi_j = invJ * basis[j].derive();
    auto phi_i = basis[i];
    return (phi_i * NablaPhi_j.dot(b_))(quadrature_point);
  }
};

// template argument deduction guide
template <int L> Gradient(const SVector<L>&) -> Gradient<L>;

// out of class definition for dot product, allow for formal syntax dot(b, Gradient()) where b is any vector
template <int L>
Gradient<L> dot(const SVector<L>& b, const Gradient<0>& g){
  return Gradient(b);
}

#endif // __GRADIENT_H__
