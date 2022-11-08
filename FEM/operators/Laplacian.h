#ifndef __LAPLACIAN_H__
#define __LAPLACIAN_H__

#include "../../utils/Symbols.h"
#include "../../utils/fields/VectorField.h"
#include <type_traits>
using fdaPDE::core::VectorField;
#include "../../utils/fields/MatrixField.h"
using fdaPDE::core::MatrixField;
#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;
#include "BilinearFormExpressions.h"
using fdaPDE::core::FEM::BilinearFormExpr;

namespace fdaPDE{
namespace core{
namespace FEM{

  // class representing the laplacian operator (isotropic and anisotropic diffusion)
  template <typename T = DefaultOperator>
  class Laplacian : public BilinearFormExpr<Laplacian<T>>{
    // perform compile-time sanity checks
    static_assert(std::is_same<DefaultOperator, T>::value ||        // implicitly set K_ = I
		  std::is_base_of<MatrixBase, T>::value ||          // space-varying case
		  std::is_base_of<Eigen::MatrixBase<T>, T>::value); // constant coefficient case
  private:
    T K_; // diffusion tensor (either constant or space-varying)
  public: 
    // constructors
    Laplacian() = default; // use default constructor for isotropic diffusion
    Laplacian(const T& K) : K_(K) {}

    std::tuple<Laplacian<T>> getTypeList() const { return std::make_tuple(*this); }
    static constexpr bool is_space_varying = std::is_base_of<MatrixBase, T>::value;
    
    // approximates the contribution to the (i,j)-th element of the discretization matrix given by the transport term:
    // \int_e \Nabla phi_i.dot(\Nabla phi_j)
    // basis: any type compliant with a functional basis behaviour. See LagrangianBasis.h for an example
    //        NOTE: we assume "basis" to provide functions already defined on the reference element
    // e: the element on which we are integrating
    // i,j: indexes of the discretization matrix element we are computing

    // NOTE: is important to use auto return type to let the compiler return the whole expression template produced by this
    // operator avoiding both type erause (e.g. by casting to some ScalarField object) as well as the creation of temporaries
    template <unsigned int M, unsigned int N, unsigned int R, typename B>
    auto integrate(const B& basis, const Element<M, N, R>& e, int i , int j) const{
      // express gradient of basis function over e in terms of gradient of basis function defined over the reference element.
      // This entails to compute (J^{-1})^T * \Nabla phi_i.
      Eigen::Matrix<double, N, M> invJ = e.invBarycentricMatrix().transpose(); // (J^{-1})^T = invJ     
      if constexpr(std::is_same<DefaultOperator, T>::value)
	// isotropic unitary diffusion fallback to K_ = I: \Nabla phi_i.dot(\Nabla * phi_j)
	return (invJ*basis[i].derive()).dot(invJ*basis[j].derive());
      else
	// anisotropic diffusion: (\Nabla phi_i)^T * K * \Nabla * phi_j
	return (invJ*basis[i].derive()).dot(K_*(invJ*basis[j].derive()));
    }
  };
  
  // template argument deduction guide
  template <typename T> Laplacian(const T&) -> Laplacian<T>;

}}}
#endif // __LAPLACIAN_H__
