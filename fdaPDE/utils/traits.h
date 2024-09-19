// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_TRAITS_H__
#define __FDAPDE_TRAITS_H__

#include <tuple>
#include <type_traits>

#include "assert.h"

namespace fdapde {

// deduces the return type of the subscript operator with arguments Args... applied to type T
template <typename T, typename... Args> struct subscript_result_of {
    using type = decltype(std::declval<T>().operator[](std::declval<Args>()...));
};

template <typename Fn, typename Arg>
concept is_subscriptable = requires(Fn fn, Arg arg) {
    { fn.operator[](arg) };
};

// trait to detect if a type is a base of a template
template <template <typename...> typename B, typename D> struct is_base_of_template {
    using U = typename std::decay<D>::type;
    // valid match (derived-to-base conversion applies)
    template <typename... Args> static std::true_type test(B<Args...>&);
    // any other match is false (D cannot be converted to its base type B)
    static std::false_type test(...);
    static constexpr bool value = decltype(test(std::declval<U&>()))::value;
};

// returns true if type T is instance of template E<F> with F some type.
template <typename T, template <typename...> typename E> struct is_instance_of : std::false_type { };
template <typename... T, template <typename...> typename E>   // valid match
struct is_instance_of<E<T...>, E> : std::true_type { };

// metaprogramming routines for working on std::tuple<>-based typelists

// returns true if std::tuple contains type T
template <typename T, typename Tuple> struct has_type { };
template <typename T> struct has_type<T, std::tuple<>> : std::false_type { };
template <typename T, typename U, typename... Args>
struct has_type<T, std::tuple<U, Args...>> : has_type<T, std::tuple<Args...>> { };
template <typename T, typename... Args> struct has_type<T, std::tuple<T, Args...>> : std::true_type { };
  
template <typename T, typename Tuple> static constexpr bool has_type_v = has_type<T, Tuple>::value;

// returns true if std::tuple contains an instantiation of template E<F>
template <template <typename, typename...> typename E, typename Tuple> struct has_instance_of { };
template <template <typename, typename...> typename E> struct has_instance_of<E, std::tuple<>> : std::false_type { };
template <typename F, template <typename, typename...> typename E, typename... Tail>
struct has_instance_of<E, std::tuple<E<F>, Tail...>> : std::true_type { };
template <typename U, template <typename, typename...> typename E, typename... Tail>   // recursive step
struct has_instance_of<E, std::tuple<U, Tail...>> {
    static constexpr bool value = has_instance_of<E, std::tuple<Tail...>>::value;
};
  
template <template <typename, typename...> typename E, typename Tuple>
static constexpr bool has_instance_of_v = has_instance_of<E, Tuple>::value;

// trait to detect whether all types in a parameter pack are unique
template <typename... Ts> struct unique_types;
// consider a pair of types and develop a tree of matches starting from them
template <typename T1, typename T2, typename... Ts> struct unique_types<T1, T2, Ts...> {
    static constexpr bool value =
      unique_types<T1, T2>::value && unique_types<T1, Ts...>::value && unique_types<T2, Ts...>::value;
};
template <typename T1, typename T2> struct unique_types<T1, T2> {
    static constexpr bool value = !std::is_same<T1, T2>::value;
};
template <typename T1> struct unique_types<T1> : std::true_type { };

template <typename... Ts> static constexpr bool unique_types_v = unique_types<Ts...>::value;

// detect if all types in a pack are equal to a given type
template <typename T, typename... Ts> struct is_all_same;
template <typename T, typename T1, typename... Ts> struct is_all_same<T, T1, Ts...> {
    static constexpr bool value = std::is_same<T, T1>::value && is_all_same<T, Ts...>::value;
};
template <typename T> struct is_all_same<T> : std::true_type { };
  
template <typename T, typename... Ts> static constexpr bool is_all_same_v = is_all_same<T, Ts...>::value;
  
// trait to detect wheter all types in a parameter pack are the same type
template <typename... Ts> struct equal_types;
template <typename T1, typename... Ts> struct equal_types<T1, Ts...> {
    static constexpr bool value = is_all_same_v<T1, Ts...>;
};

template <typename... Ts> static constexpr bool equal_types_v = equal_types<Ts...>::value;

// obtain index of type in tuple (assume types are unique in the std::tuple)
template <typename T, typename tuple> struct index_of_type;
template <typename T, typename... Ts>
    requires(unique_types_v<Ts...>)
struct index_of_type<T, std::tuple<Ts...>> {
   private:
    template <std::size_t... idx> static constexpr int find_idx(std::index_sequence<idx...>) {
        return -1 + ((std::is_same<T, Ts>::value ? idx + 1 : 0) + ... + 0);
    }
   public:
    static constexpr int index = find_idx(std::index_sequence_for<Ts...> {});
};

// evaluate metafunction based on condition
template <bool b, typename T, typename F> struct eval_if : std::type_identity<typename T::type> { };
template <typename T, typename F> struct eval_if<false, T, F> : std::type_identity<typename F::type> { };

// a compile time switch for selecting between multiple types based on condition
template <bool b, typename T> struct switch_type_case {   // case type
    static constexpr bool value = b;
    using type = T;
};
template <typename SwitchCase, typename... Ts> struct switch_type {
    using type = typename eval_if<SwitchCase::value, SwitchCase, switch_type<Ts...>>::type;
};
template <typename SwitchCase> struct switch_type<SwitchCase> {   // end of recursion
    fdapde_static_assert(SwitchCase::value, NO_TRUE_CONDITION_IN_SWITCH_TYPE_STATEMENT);
    using type = typename SwitchCase::type;
};

// macro for the definition of has_x detection idiom traits
#define define_has(METHOD)                                                                                             \
    template <typename T, typename sig, typename = void> struct has_##METHOD : std::false_type { };                    \
    template <typename T, typename... Args>                                                                            \
    struct has_##METHOD<T, void(Args...), std::void_t<decltype(std::declval<T>().METHOD(std::declval<Args>()...))>> :  \
        std::true_type { };                                                                                            \
    template <typename T, typename R, typename... Args>                                                                \
    struct has_##METHOD<                                                                                               \
      T, R(Args...),                                                                                                   \
      typename std::enable_if<                                                                                         \
        !std::is_void<R>::value &&                                                                                     \
        std::is_convertible<decltype(std::declval<T>().METHOD(std::declval<Args>()...)), R>::value>::type> :           \
        std::true_type { };

// usefull member function pointers trait
template <typename F> struct fn_ptr_traits_base { };
template <typename R, typename T, typename... Args> struct fn_ptr_traits_base<R (T::*)(Args...)> {
    using RetType = R;
    using ArgsType = std::tuple<Args...>;
    static constexpr int n_args = sizeof...(Args);
    using ClassType = T;
    using FnPtrType = R (*)(void*, Args&&...);    // void* is the pointer to the object instance
};
template <typename F> struct fn_ptr_traits_impl { };
template <typename R, typename T, typename... Args>
struct fn_ptr_traits_impl<R (T::*)(Args...)> : public fn_ptr_traits_base<R (T::*)(Args...)> {
    using MemFnPtrType = R (T::*)(Args...);
};
template <typename R, typename T, typename... Args>
struct fn_ptr_traits_impl<R (T::*)(Args...) const> : public fn_ptr_traits_base<R (T::*)(Args...)> {
    using MemFnPtrType = R (T::*)(Args...) const;
};
template <auto FnPtr> struct fn_ptr_traits : public fn_ptr_traits_impl<decltype(FnPtr)> { };

// trait to detect if T is an Eigen dense matrix
template <typename T> struct is_eigen_dense {
  static constexpr bool value = std::is_base_of<Eigen::MatrixBase<T>, T>::value;
};
template <typename T> constexpr bool is_eigen_dense_v = is_eigen_dense<T>::value;
  
// trait to detect if T is an Eigen vector
template <typename T> class is_eigen_dense_vector {
   private:
    static constexpr bool check_() {
        if constexpr (std::is_base_of<Eigen::MatrixBase<T>, T>::value) {
            if constexpr (T::ColsAtCompileTime == 1) return true;
            return false;
        }
        return false;
    }
   public:
    static constexpr bool value = check_();
};
template <typename T> constexpr bool is_eigen_dense_vector_v = is_eigen_dense_vector<T>::value;

}   // namespace fdapde

#endif   // __FDAPDE_TRAITS_H__
