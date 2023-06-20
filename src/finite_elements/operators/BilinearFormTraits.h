#ifndef __BILINEAR_FORM_TRAITS_H__
#define __BILINEAR_FORM_TRAITS_H__

#include <type_traits>
#include <tuple>
#include "../../utils/Traits.h"
#include "Gradient.h"
using fdaPDE::core::FEM::Gradient;
#include "Identity.h"
using fdaPDE::core::FEM::Identity;
#include "Laplacian.h"
using fdaPDE::core::FEM::Laplacian;
#include "dT.h"
using fdaPDE::core::FEM::dT;

namespace fdaPDE{
namespace core{
namespace FEM{

  // trait to detect if the bilinear form is symmetric. 
  template <typename E> struct is_symmetric {
    // returns false if an instantiation of Gradient<> is contained in the typelist of the bilinar form
    static constexpr bool value = !has_instance_of<Gradient, decltype(std::declval<E>().getTypeList())>::value;
  };

  // trait to detect if the bilinear form denotes an elliptic operator.
  template <typename E> struct is_elliptic {
  public:
    // returns true if an instantiation of Gradient<> is not contained in the typelist of the bilinar form
    static constexpr bool value =
      has_instance_of<Laplacian, decltype(std::declval<E>().getTypeList())>::value ||
      has_instance_of<Identity,  decltype(std::declval<E>().getTypeList())>::value ||
      has_instance_of<Gradient,  decltype(std::declval<E>().getTypeList())>::value;
  };

  // trait to detect if the bilinear form denotes a parabolic PDE.
  template <typename E> struct is_parabolic {
    // returns true if the time derivative operator dT() is detected in the expression
    static constexpr bool value = has_instance_of<dT, decltype(std::declval<E>().getTypeList())>::value;  
  };

}}}
#endif // __BILINEAR_FORM_TRAITS_H__
