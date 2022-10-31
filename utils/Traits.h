#ifndef __TRAITS_H__
#define __TRAITS_H__

#include <type_traits>

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
  
};

#endif // __TRAITS_H__
