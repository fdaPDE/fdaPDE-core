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
    
    // approximates the contribution to the (i,j)-th element of the discretization matrix given by the diffusion term:

    // NOTE: is important to use auto return type to let the compiler return the whole expression template produced by this
    // operator avoiding both type erause (e.g. by casting to some ScalarField object) as well as the creation of temporaries
    template <typename... Args>
    auto integrate(const std::tuple<Args...>& mem_buffer) const {
      IMPORT_MEM_BUFFER_SYMBOLS(mem_buffer);
      // express gradient of basis function over e in terms of gradient of basis function defined over the reference element.
      // This entails to compute (J^{-1})^T * \Nabla phi_i.
      if constexpr(std::is_same<DefaultOperator, T>::value)
	// isotropic unitary diffusion fallback to K_ = I: \Nabla phi_i.dot(\Nabla * phi_j)
	return (invJ*NablaPsi_i).dot(invJ*NablaPsi_j);
      else
	// anisotropic diffusion: (\Nabla phi_i)^T * K * \Nabla * phi_j
	return (invJ*NablaPsi_i).dot(K_*(invJ*NablaPsi_j));
    }
  };
  
  // template argument deduction guide
  template <typename T> Laplacian(const T&) -> Laplacian<T>;

}}}
#endif // __LAPLACIAN_H__
