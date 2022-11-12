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
    
    // approximates the contribution of this operator for the (i,j)-th element of the discretization matrix
    // IMPORT_MEM_BUFFER_SYMBOLS makes the proper unpack of the mem_buffer tuple by introducing a set of symbols,
    // symbols are set via field pointers by the assembly loop. See BilinearFormExpressions.h for its definition
    template <typename... Args>
    auto integrate(const std::tuple<Args...>& mem_buffer) const {
      IMPORT_MEM_BUFFER_SYMBOLS(mem_buffer);
      if constexpr(std::is_same<DefaultOperator, T>::value)
	// isotropic unitary diffusion fallback to K_ = I: (\Nabla psi_i).dot(\Nabla psi_j)
	return (invJ*NablaPsi_i).dot(invJ*NablaPsi_j);
      else
	// non unitary or anisotropic diffusion: (\Nabla psi_i)^T*K*(\Nabla \psi_j)
	return (invJ*NablaPsi_i).dot(K_*(invJ*NablaPsi_j));
    }
  };
  
  // template argument deduction guide
  template <typename T> Laplacian(const T&) -> Laplacian<T>;

}}}
#endif // __LAPLACIAN_H__
