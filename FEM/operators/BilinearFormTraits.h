#ifndef __BILINEAR_FORM_TRAITS_H__
#define __BILINEAR_FORM_TRAITS_H__

#include <type_traits>
#include <tuple>

#include "Gradient.h"
#include "Identity.h"
#include "Laplacian.h"
#include "dT.h"

// returns true if type T is instance of template E<L> with L some unsigned int.
template <typename T, template <unsigned int L> typename E>
struct is_instance_of : std::false_type {};
template <template<unsigned int> typename E, unsigned int L> // valid match
struct is_instance_of<E<L>, E> : std::true_type {};

// returns std::true_type if tuple contains type T
template <typename T, typename Tuple> struct has_type {};
// an empty tuple cannot contain T, return false
template <typename T> struct has_type<T, std::tuple<>>
  : std::false_type {};
// if the head of the tuple is not of type T, go on recursively on the remaining types
template <typename T, typename U, typename... Args>
struct has_type<T, std::tuple<U, Args...>> : has_type<T, std::tuple<Args...>> {};
// in case the head of the tuple is type T, end of recursion and return true
template <typename T, typename... Args>
struct has_type<T, std::tuple<T, Args...>>
  : std::true_type {};

// returns std::true_type if tuple contains an instantiation of template E<L>
template <template <unsigned int L> typename E, typename Tuple> struct has_instance_of {};

template <template <unsigned int L> typename E> // empty tuple cannot contain anything
struct has_instance_of<E, std::tuple<>> {
  using type = std::false_type;
};

template <unsigned int L, template <unsigned int> typename E, typename... Tail> // type found, stop recursion
struct has_instance_of<E, std::tuple<E<L>, Tail...>> {
  using type = std::true_type;
};

template <typename U, template <unsigned int L> typename E, typename... Tail>   // recursive step
struct has_instance_of<E, std::tuple<U, Tail...>> {
  using type = typename has_instance_of<E, std::tuple<Tail...>>::type;
};

// trait to detect if the bilinear form is symmetric. 
template <typename E> struct is_symmetric {
public:
  // returns false if an instantiation of Gradient<> is contained in the typelist of the bilinar form
  static constexpr bool value = !has_instance_of<Gradient, decltype(std::declval<E>().getTypeList())>::type::value;
};

// trait to detect if the bilinear form denotes an elliptic operator.
template <typename E> struct is_elliptic {
public:
  // returns true if an instantiation of Gradient<> is not contained in the typelist of the bilinar form
  static constexpr bool value =
    has_instance_of<Laplacian, decltype(std::declval<E>().getTypeList())>::type::value ||
    has_instance_of<Identity,  decltype(std::declval<E>().getTypeList())>::type::value ||
    has_instance_of<Gradient,  decltype(std::declval<E>().getTypeList())>::type::value;
};

// trait to detect if the bilinear form denotes a parabolic PDE.
template <typename E> struct is_parabolic {
public:
  // returns true if the time derivative operator dT() is detected in the expression
  static constexpr bool value = has_instance_of<dT, decltype(std::declval<E>().getTypeList())>::type::value;  
};

#endif // __BILINEAR_FORM_TRAITS_H__
