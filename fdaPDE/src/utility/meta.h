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

#ifndef __FDAPDE_META_H__
#define __FDAPDE_META_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// apply lambda F_ to each value in index pack {0, ..., N_ - 1}
template <int N_, typename F_> constexpr decltype(auto) apply_index_pack(F_&& f) {
    return [&]<int... Ns_>(std::integer_sequence<int, Ns_...>) -> decltype(auto) {
        return f.template operator()<Ns_...>();
    }(std::make_integer_sequence<int, N_> {});
}
template <int N_, typename F_> constexpr void for_each_index_in_pack(F_&& f) {
    [&]<int... Ns_>(std::integer_sequence<int, Ns_...>) {
        (f.template operator()<Ns_>(), ...);
    }(std::make_integer_sequence<int, N_> {});
}

// apply lambda F_ to each index and args pair {(0, args[0]), ..., (N_ - 1, args[N_ - 1])}
template <int N_, typename F_, typename... Args_>
    requires(sizeof...(Args_) == N_)
constexpr void for_each_index_and_args(F_&& f, Args_&&... args) {
    [&]<int... Ns_>(std::integer_sequence<int, Ns_...>) {
        (f.template operator()<Ns_, Args_>(std::forward<Args_>(args)), ...);
    }(std::make_integer_sequence<int, N_> {});
}

// a tuple of pairs {{0, T_0}, {1, T_1}, ..., {N, T_N}}
template <typename... Ts> struct indexed_tuple {
   private:
    template <std::size_t N_, typename T_>
        requires(std::is_default_constructible_v<T_>)
    struct pair_t {
        static constexpr int index = N_;
        using type = T_;

        pair_t() : value() { }
        pair_t(const T_& value_) : value(value_) { }
        T_ value;
    };
    template <typename IdxList> struct idx_list;
    template <std::size_t... Idxs> struct idx_list<std::index_sequence<Idxs...>> {
        using type = std::tuple<pair_t<Idxs, Ts>...>;
    };
   public:
    using type = typename idx_list<std::make_index_sequence<sizeof...(Ts)>>::type;

    indexed_tuple(const Ts&... ts) : value(ts...) { }
    type value;
};
template <typename... Ts> auto make_indexed_tuple(Ts&&... ts) { return indexed_tuple(std::forward<Ts>(ts)...).value; }
// apply functor F to tuple of indexed pairs
template <int N_, typename F_, typename... Args_>
    requires(sizeof...(Args_) == N_)
constexpr decltype(auto) apply_index_pack_and_args(F_&& f, Args_&&... args) {
    return [&]<typename... Ts_>(const std::tuple<Ts_...>& tuple) -> decltype(auto) {
        return f.template operator()<Ts_...>(std::get<Ts_::index>(tuple).value...);
    }(make_indexed_tuple(args...));
}

// detect if type is integer-like
template <typename T> class is_integer {
    using T_ = std::decay_t<T>;
   public:
    static constexpr bool value =
      std::is_arithmetic_v<T_> && !std::is_same_v<T_, bool> && !std::is_floating_point_v<T_>;
};
template <typename T> static constexpr bool is_integer_v = is_integer<T>::value;

// detect if T is a std::tuple<> instance
template <typename T> struct is_tuple : std::false_type { };
template <typename... Ts> struct is_tuple<std::tuple<Ts...>> : std::true_type { };
template <typename T1, typename T2>   // std::pair detected as tuple instance
struct is_tuple<std::pair<T1, T2>> : std::true_type { };
template <typename T> static constexpr bool is_tuple_v = is_tuple<std::decay_t<T>>::value;
  
// detect if T is a pair-like object
template <typename T>
struct is_pair {
    static constexpr bool value = []() {
        if constexpr (is_tuple_v<std::decay_t<T>>) {
            if constexpr (std::tuple_size_v<std::decay_t<T>> == 2) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }();
};
template <typename T> static constexpr bool is_pair_v = is_pair<T>::value;

// detect if T is a shared_ptr instance
template <typename T> struct is_shared_ptr : std::false_type { };
template <typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
template <typename T> static constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;
  // given a type E, returns T if E = std::shared_ptr<T>, or directly returns E otherwise
template <typename E> struct remove_shared_ptr {
    using type = decltype([]() {
      if constexpr (is_shared_ptr_v<std::decay_t<E>>) {
          return typename E::element_type();
        } else {
            return E();
        }
    }());
};
template <typename E> using remove_shared_ptr_t = typename remove_shared_ptr<E>::type;

template <typename Fn, typename... Arg>
concept is_subscriptable = requires(Fn fn, Arg... arg) {
    { fn.operator[](arg...) };
};
template <typename T, typename... Args>
    requires(is_subscriptable<T, Args...>)
struct subscript_result_of {
    using type = decltype(std::declval<T>().operator[](std::declval<Args>()...));
};
// deduces returned type of T's subscript operator with arguments Args...
template <typename T, typename... Args> using subscript_result_of_t = typename subscript_result_of<T, Args...>::type;

// detects if T is callable with Order arguments of type IndexT
template <typename T, int Order, typename IndexT> class is_indexable {
    using T_ = std::decay_t<T>;
   public:
    static constexpr bool value = []() {
        return internals::apply_index_pack<Order>([]<int... Ns_>() {
            if constexpr (requires(T_ t) { t(((void)Ns_, IndexT())...); }) {
                return true;
            } else {
                return false;
            }
        });
    }();
};
template <typename T, int Order, typename IndexT>
static constexpr bool is_indexable_v = is_indexable<T, Order, IndexT>::value;
  
// detects if T behaves like a vector
template <typename T> class is_vector_like {
    using T_ = std::decay_t<T>;
   public:
    static constexpr bool value = []() {
#ifdef __FDAPDE_HAS_EIGEN__
        if constexpr (internals::is_eigen_dense_xpr_v<T_>) {
            return internals::is_eigen_dense_vec_v<T_>;
        } else
#endif
            return (is_subscriptable<T_, int> || (is_indexable_v<T_, 1, int> && !is_indexable_v<T_, 2, int>)) &&
                   requires(T_ t) {
                       { t.size() } -> std::convertible_to<int>;
                   };
    }();
};
template <typename T> static constexpr bool is_vector_like_v = is_vector_like<T>::value;

template <typename T>
    requires(is_vector_like_v<T>)
decltype(auto) vector_like_access(T&& data, int index) {
    using T_ = std::decay_t<T>;
    if constexpr (is_subscriptable<T_, int>) {
        return data[index];
    } else {
        return data(index);
    }
}
  
// detect if T behaves like a matrix
template <typename T> class is_matrix_like {
    using T_ = std::decay_t<T>;
  public:
   static constexpr bool value = internals::is_indexable_v<T_, 2, int> && requires(T_ t) {
       { t.rows() } -> std::convertible_to<int>;
       { t.cols() } -> std::convertible_to<int>;
   };
};
template <typename T> static constexpr bool is_matrix_like_v = is_matrix_like<T>::value;

// get i-th element from parameter pack
template <int N, typename... Ts>
    requires(sizeof...(Ts) >= N)
struct pack_element : std::type_identity<std::tuple_element_t<N, std::tuple<Ts...>>> { };
template <int N, typename... Ts> using pack_element_t = typename pack_element<N, Ts...>::type;
  
// detect whether there are no duplicate types in pack
template <typename... Ts> struct unique;
template <typename T1, typename T2, typename... Ts> struct unique<T1, T2, Ts...> {
    static constexpr bool value = unique<T1, T2>::value && unique<T1, Ts...>::value && unique<T2, Ts...>::value;
};
template <typename T1, typename T2> struct unique<T1, T2> {
    static constexpr bool value = !std::is_same_v<T1, T2>;
};
template <typename T1> struct unique<T1> : std::true_type { };
template <typename... Ts> constexpr bool unique_v = unique<Ts...>::value;

// detect if all types in pack are equal to T
template <typename T, typename... Ts> struct all_same;
template <typename T, typename T1, typename... Ts> struct all_same<T, T1, Ts...> {
    static constexpr bool value = std::is_same<T, T1>::value && all_same<T, Ts...>::value;
};
template <typename T> struct all_same<T> : std::true_type { };
template <typename T, typename... Ts> static constexpr bool all_same_v = all_same<T, Ts...>::value;
  
// detect wheter all types in pack are equal
template <typename... Ts> struct all_equal;
template <typename T1, typename... Ts> struct all_equal<T1, Ts...> {
    static constexpr bool value = all_same_v<T1, Ts...>;
};
template <typename... Ts> static constexpr bool all_equal_v = all_equal<Ts...>::value;

// index of type in tuple
template <typename T, typename Tuple> struct index_of;
template <typename T, typename... Ts> struct index_of<T, std::tuple<Ts...>> {
   private:
    template <std::size_t... idx> static constexpr int find_idx(std::index_sequence<idx...>) {
        bool found = false;
        int count = 0;
        ((!found ? (++count, found = std::is_same_v<T, Ts>) : 0) + ...);
        return found ? count - 1 : -1;
    }
   public:
    static constexpr int value = find_idx(std::index_sequence_for<Ts...> {});
};

// returns std::true_type if tuple contains type T
template <typename T, typename Tuple> struct has_type { };
template <typename T> struct has_type<T, std::tuple<>> : std::false_type { };
template <typename T, typename U, typename... Args> struct has_type<T, std::tuple<U, Args...>> {
    static constexpr bool value = has_type<T, std::tuple<Args...>>::value;
};
template <typename T, typename... Args> struct has_type<T, std::tuple<T, Args...>> {
    static constexpr bool value = true;
};

// detect if T is same to at least one of the types in Tuple
template <typename T, typename Tuple> struct is_any_same {
    static constexpr bool value = has_type<T, Tuple>::value;
};
template <typename T, typename Tuple> static constexpr bool is_any_same_v = is_any_same<T, Tuple>::value;

// heterogeneous value list
template <auto... Vs> struct value_list { };

// size of value list
template <typename T> struct size { };
template <auto... Vs> struct size<value_list<Vs...>> {
    static constexpr int value = sizeof...(Vs);
};
// head of value list
template <typename T> struct head { };
template <auto V1, auto... Vs> struct head<value_list<V1, Vs...>> : std::type_identity<decltype(V1)> { };
// tail of value list
template <typename T> struct tail { };
template <auto V1, auto V2, auto... Vs>
struct tail<value_list<V1, V2, Vs...>> : std::type_identity<typename tail<value_list<V2, Vs...>>::type> { };
template <auto V1, auto V2> struct tail<value_list<V1, V2>> : std::type_identity<decltype(V2)> { };
template <auto V1> struct tail<value_list<V1>> : std::type_identity<decltype(V1)> { };
// value list merge
template <typename V1, typename V2> struct merge_impl { };
template <auto... Vs1, auto... Vs2> struct merge_impl<value_list<Vs1...>, value_list<Vs2...>> {
    using type = value_list<Vs1..., Vs2...>;
};
template <typename... Vs> struct merge { };
template <typename V1, typename V2, typename... Vs> struct merge<V1, V2, Vs...> {
    using type = typename merge<typename merge_impl<V1, V2>::type, Vs...>::type;
};
template <typename V> struct merge<V> : std::type_identity<V> { };

// evaluate metafunction based on condition
template <bool b, typename T, typename F> struct eval_if : std::type_identity<typename T::type> { };
template <typename T, typename F> struct eval_if<false, T, F> : std::type_identity<typename F::type> { };

// compile time switch for selecting between multiple types based on condition
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
  
// member function pointers trait
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

// if XprType has its NestAsRef bit set, returns the type XprType&, otherwise return XprType.
template <typename XprType>
concept has_nest_as_ref_bit = requires(XprType t) { XprType::NestAsRef; };

template <typename XprType, bool v> struct ref_select_impl;
template <typename XprType> struct ref_select_impl<XprType, true> {
    using type = std::conditional_t<
      XprType::NestAsRef == 0, std::remove_reference_t<XprType>, std::add_lvalue_reference_t<XprType>>;
};
template <typename XprType> struct ref_select_impl<XprType, false> {
    using type = XprType;
};
template <typename XprType> struct ref_select : ref_select_impl<XprType, has_nest_as_ref_bit<XprType>> { };
template <typename XprType> using ref_select_t = ref_select<XprType>::type;

// selects one between arg1 and arg2 based on condition f
template <typename Arg1, typename Arg2, typename F>
    requires(
      requires(F f, Arg1 arg1, Arg2 arg2) {
          { f(arg1, arg2) } -> std::same_as<bool>;
      } ||
      requires(F f) {
          { f() } -> std::same_as<bool>;
      })
const auto& select_one_between(const Arg1& arg1, const Arg2& arg2, F&& f) {
    if constexpr ([&]() {
                      if constexpr (requires(F f, Arg1 arg1, Arg2 arg2) {
                                        { f(arg1, arg2) } -> std::same_as<bool>;
                                    }) {
                          return f(arg1, arg2);
                      } else {
                          return f();
                      }
                  }()) {
        return arg1;
    } else {
        return arg2;
    }
}

// check if two types are related by inheritance
template <typename T, typename W> struct are_related_by_inheritance {
    static constexpr bool value = std::is_base_of_v<T, W> || std::is_base_of_v<W, T>;
};
template <typename T, typename W> constexpr bool are_related_by_inheritance_v = are_related_by_inheritance<T, W>::value;

// returns the most derived type between T and W, or T otherwise
template <typename T, typename W> struct prefer_most_derived {
    using type = std::conditional_t<std::is_same_v<T, W>, T, std::conditional_t<std::is_base_of_v<T, W>, W, T>>;
};
template <typename T, typename W> using prefer_most_derived_t = typename prefer_most_derived<T, W>::type;
  
}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_META_H__
