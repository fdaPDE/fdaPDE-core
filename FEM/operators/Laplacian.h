#ifndef __LAPLACIAN_H__
#define __LAPLACIAN_H__

#include "../../utils/Symbols.h"
#include "../../utils/fields/VectorField.h"
using fdaPDE::core::VectorField;
#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;
#include "BilinearFormExpressions.h"
using fdaPDE::core::FEM::BilinearFormExpr;

namespace fdaPDE{
namespace core{
namespace FEM{

  // class representing the laplacian operator (isotropic and anisotropic diffusion)
  template <unsigned int L = 0>
  class Laplacian : public BilinearFormExpr<Laplacian<L>>{
  private:
    SMatrix<L> diffusionTensor_{};
  public:
    // constructors
    Laplacian() = default; // use default constructor for isotropic diffusion
    Laplacian(const SMatrix<L>& diffusionTensor) : diffusionTensor_(diffusionTensor) {}

    std::tuple<Laplacian<L>> getTypeList() const { return std::make_tuple(*this); }
    
    // approximates the contribution to the (i,j)-th element of the discretization matrix given by the transport term:
    // \int_e \Nabla phi_i.dot(\Nabla phi_j)
    // basis: any type compliant with a functional basis behaviour. See LagrangianBasis.h for an example
    //        NOTE: we assume "basis" to provide functions already defined on the reference element
    // e: the element on which we are integrating
    // i,j: indexes of the discretization matrix element we are computing
   template <unsigned int M, unsigned int N, unsigned int R, typename B>
   ScalarField<M> integrate(const B& basis, const Element<M, N, R>& e, int i , int j) const{
     // express gradient of basis function over e in terms of gradient of basis function defined over the reference element.
     // This entails to compute (J^{-1})^T * \Nabla phi_i.
     Eigen::Matrix<double, N, M> invJ = e.invBarycentricMatrix().transpose(); // (J^{-1})^T = invJ
     VectorField<M,N> NablaPhi_i = invJ * basis[i].derive(); 
     VectorField<M,N> NablaPhi_j = invJ * basis[j].derive();
     
     if constexpr (L == 0) // for isotropic laplacian:          \Nabla phi_i.dot(\Nabla * phi_j)
       return NablaPhi_i.dot(NablaPhi_j);
     if constexpr (L != 0){ // case for anisotropic diffusion: (\Nabla phi_i)^T * K * \Nabla * phi_j
       return NablaPhi_i.dot(diffusionTensor_*NablaPhi_j);
     }
   }
  };
  
  // template argument deduction guide
  template <int L> Laplacian(const SMatrix<L>&) -> Laplacian<L>;

}}}
#endif // __LAPLACIAN_H__
