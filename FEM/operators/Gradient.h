#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#include "../../utils/Symbols.h"
#include "../../utils/fields/VectorField.h"
#include <cstddef>
#include <type_traits>
using fdaPDE::core::VectorField;
#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;
#include "BilinearFormExpressions.h"
using fdaPDE::core::FEM::BilinearFormExpr;

namespace fdaPDE{
namespace core{
namespace FEM{

  // constexpr function to detect if T is an Eigen vector
  template <typename T>
  bool constexpr is_eigen_vector() {
    // check if T is an eigen matrix
    if constexpr(std::is_base_of<Eigen::MatrixBase<T>, T>::value)
      // hide access to ::ColsAtCompileTime to all types T which are not Eigen matrices
      return T::ColsAtCompileTime == 1; // has T exactly one column?
    return false;
  };
  
  // A class to provide the discretization for the gradient operator (transport term).
  template <typename T>
  class Gradient : public BilinearFormExpr<Gradient<T>>{
    // perform compile-time sanity checks
    static_assert(std::is_base_of<VectBase, T>::value || // space-varying case
		  is_eigen_vector<T>());                 // constant coefficient case
  private:
    T b_; // transport vector (either constant or space-varying)
  public:
    // constructors
    Gradient(const T& b) : b_(b) {}

    // compile time informations
    std::tuple<Gradient<T>> getTypeList() const { return std::make_tuple(*this); }
    static constexpr bool is_space_varying = std::is_base_of<VectBase, T>::value;
    
    // approximates the contribution to the (i,j)-th element of the discretization matrix given by the transport term:
    // \int_e phi_i * b.dot(\Nabla phi_j)
    // basis: any type compliant with a functional basis behaviour. See LagrangianBasis.h for an example
    //        NOTE: we assume "basis" to provide functions already defined on the reference element
    // e: the pyhsical element on which we are integrating
    // i,j: indexes of the discretization matrix entry we are computing

    // NOTE: is important to use auto return type to let the compiler return the whole expression template produced by this
    // operator avoiding both type erause (e.g. by casting to some ScalarField object) as well as the creation of temporaries
    template <unsigned int M, unsigned int N, unsigned int R, typename B>
    auto integrate(const B& basis, const Element<M, N, R>& e, int i , int j) const{
      // express gradient of basis function over e in terms of gradient of basis function defined over the reference element.
      // This entails to compute (J^{-1})^T * \Nabla phi_i.
      Eigen::Matrix<double, N, M> invJ = e.invBarycentricMatrix().transpose(); // (J^{-1})^T = invJ
      return basis[i]*(invJ*basis[j].derive()).dot(b_);
    }
  };  
  // template argument deduction guide
  template <typename T> Gradient(const T&) -> Gradient<T>;
  
}}}
#endif // __GRADIENT_H__
