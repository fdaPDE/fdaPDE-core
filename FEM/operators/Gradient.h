#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#include <cstddef>
#include <type_traits>
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
    static_assert(std::is_base_of<VectorBase, T>::value || // space-varying case
		  is_eigen_vector<T>());                   // constant coefficient case
  private:
    T b_; // transport vector (either constant or space-varying)
  public:
    // constructors
    Gradient(const T& b) : b_(b) {}

    // compile time informations
    std::tuple<Gradient<T>> getTypeList() const { return std::make_tuple(*this); }
    static constexpr bool is_space_varying = std::is_base_of<VectorBase, T>::value;
    
    // approximates the contribution of this operator for the (i,j)-th element of the discretization matrix
    // IMPORT_MEM_BUFFER_SYMBOLS makes the proper unpack of the mem_buffer tuple by introducing a set of symbols,
    // symbols are set via field pointers by the assembly loop. See BilinearFormExpressions.h for its definition
    template <typename... Args>
    auto integrate(const std::tuple<Args...>& mem_buffer) const {
      IMPORT_MEM_BUFFER_SYMBOLS(mem_buffer);
      return psi_i*(invJ*NablaPsi_j).dot(b_); // \psi_i*b.dot(\nabla \psi_j)
    }
  };  
  // template argument deduction guide
  template <typename T> Gradient(const T&) -> Gradient<T>;
  
}}}
#endif // __GRADIENT_H__
