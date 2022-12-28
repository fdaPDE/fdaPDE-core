#ifndef __TRAITS_H__
#define __TRAITS_H__

#include <type_traits>
#include <tuple>

// a collection of useful traits

namespace fdaPDE{

  // trait to detect if a type is a base of a template
  template <template <typename...> typename B, typename D>
  struct is_base_of_template {
    using U = typename std::decay<D>::type;
    // valid match (derived-to-base conversion applies)
    template <typename... Args>
    static std::true_type test(B<Args...>&);
    // any other match is false (D cannot be converted to its base type B)
    static std::false_type test(...);

    static constexpr bool value = decltype(test(std::declval<U&>()))::value;
  };

  // returns true if type T is instance of template E<F> with F some type.
  template <typename T, template <typename...> typename E>
  struct is_instance_of : std::false_type {};
  template <typename... T, template <typename...> typename E> // valid match
  struct is_instance_of<E<T...>, E> : std::true_type {};
  
  // metaprogramming routines for working on std::tuple based typelists
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

  // returns std::true_type if tuple contains an instantiation of template E<F>
  template <template <typename F> typename E, typename Tuple> struct has_instance_of {};
  template <template <typename F> typename E> // empty tuple cannot contain anything
  struct has_instance_of<E, std::tuple<>> : std::false_type {};
  template <typename F, template <typename> typename E, typename... Tail> // type found, stop recursion
  struct has_instance_of<E, std::tuple<E<F>, Tail...>> : std::true_type {};
  template <typename U, template <typename> typename E, typename... Tail> // recursive step
  struct has_instance_of<E, std::tuple<U, Tail...>> {
    static constexpr bool value = has_instance_of<E, std::tuple<Tail...>>::value;
  };
  
};

#endif // __TRAITS_H__
