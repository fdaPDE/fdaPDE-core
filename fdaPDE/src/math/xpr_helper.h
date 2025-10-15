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

#ifndef __FDAPDE_XPR_HELPER_H__
#define __FDAPDE_XPR_HELPER_H__

namespace fdapde {

template <int Size, typename Derived> struct ScalarFieldBase;
template <int Size, typename Derived> struct MatrixFieldBase;
  
namespace internals {

// expression type detection traits
template <typename Xpr> struct is_scalar_field {
    static constexpr bool value = []() {
        if constexpr (requires(Xpr xpr) { Xpr::StaticInputSize; }) {
            return std::is_base_of_v<ScalarFieldBase<Xpr::StaticInputSize, Xpr>, Xpr>;
        } else {
            return false;
        }
    }();
};
template <typename Xpr> static constexpr bool is_scalar_field_v = is_scalar_field<Xpr>::value;
template <typename Xpr> struct is_matrix_field {
    static constexpr bool value = []() {
        if constexpr (requires(Xpr xpr) { Xpr::StaticInputSize; }) {
            return std::is_base_of_v<MatrixFieldBase<Xpr::StaticInputSize, Xpr>, Xpr>;
        } else {
            return false;
        }
    }();
};
template <typename Xpr> static constexpr bool is_matrix_field_v = is_matrix_field<Xpr>::value;

// expression nodes concept
template <typename T>
concept xpr_is_unary_op = requires(T t) { typename T::Derived; };
template <typename T>
concept xpr_is_binary_op = requires(T t) {
    typename T::LhsDerived;
    typename T::RhsDerived;
};
template <typename T>
concept xpr_is_leaf = !xpr_is_unary_op<T> && !xpr_is_binary_op<T>;

// checks if all leaf nodes in the expression satisfy the unary predicate UnaryPred
template <typename UnaryPred, typename Xpr> static constexpr bool xpr_all_of_impl() {
    if constexpr (!(is_scalar_field_v<Xpr> || is_matrix_field_v<Xpr>)) {
        return true;   // nodes which are not fields always evaluate true
    } else {
        if constexpr (xpr_is_leaf<Xpr>) {
            return UnaryPred {}.template operator()<Xpr>();   // check if leaf satisfies unary predicate
        } else if constexpr (xpr_is_unary_op<Xpr>) {
            return xpr_all_of_impl<UnaryPred, typename Xpr::Derived>();
        } else if constexpr (xpr_is_binary_op<Xpr>) {
            bool lhs_partial = false;
            bool rhs_partial = false;
            if constexpr (is_scalar_field_v<typename Xpr::LhsDerived> || is_matrix_field_v<typename Xpr::LhsDerived>) {
                lhs_partial = xpr_all_of_impl<UnaryPred, typename Xpr::LhsDerived>();
            }
            if constexpr (is_scalar_field_v<typename Xpr::RhsDerived> || is_matrix_field_v<typename Xpr::RhsDerived>) {
                rhs_partial = xpr_all_of_impl<UnaryPred, typename Xpr::RhsDerived>();
            }
            return lhs_partial & rhs_partial;
        }
    }
}
template <typename UnaryPred, typename Xpr> static constexpr bool xpr_all_of() {
    return xpr_all_of_impl<UnaryPred, Xpr>();
}
template <typename UnaryPred, typename Xpr>
static constexpr bool xpr_all_of([[maybe_unused]] Xpr xpr, [[maybe_unused]] UnaryPred p) {
    return xpr_all_of_impl<std::decay_t<UnaryPred>, std::decay_t<Xpr>>();
}
// checks if at least one node in the expression satisfies the unary predicate UnaryPred
template <typename UnaryPred, typename Xpr> static constexpr bool xpr_any_of_impl() {
    if constexpr (!(is_scalar_field_v<Xpr> || is_matrix_field_v<Xpr>)) {
        return true;   // nodes which are not fields always evaluate true
    } else {
        if constexpr (xpr_is_leaf<Xpr>) {
            return UnaryPred {}.template operator()<Xpr>();   // check if leaf satisfies unary predicate
        } else if constexpr (xpr_is_unary_op<Xpr>) {
            return xpr_any_of_impl<UnaryPred, typename Xpr::Derived>();
        } else if constexpr (xpr_is_binary_op<Xpr>) {
            bool lhs_partial = false;
            bool rhs_partial = false;
            if constexpr (is_scalar_field_v<typename Xpr::LhsDerived> || is_matrix_field_v<typename Xpr::LhsDerived>) {
                lhs_partial = xpr_any_of_impl<UnaryPred, typename Xpr::LhsDerived>();
            }
            if constexpr (is_scalar_field_v<typename Xpr::RhsDerived> || is_matrix_field_v<typename Xpr::RhsDerived>) {
                rhs_partial = xpr_any_of_impl<UnaryPred, typename Xpr::RhsDerived>();
            }
            return lhs_partial || rhs_partial;
        }
    }
}
template <typename UnaryPred, typename Xpr> static constexpr bool xpr_any_of() {
    return xpr_any_of_impl<UnaryPred, Xpr>();
};
template <typename UnaryPred, typename Xpr>
static constexpr bool xpr_any_of([[maybe_unused]] Xpr xpr, [[maybe_unused]] UnaryPred p) {
    return xpr_any_of_impl<std::decay_t<UnaryPred>, std::decay_t<Xpr>>();
}
  // checks if no node in the expression satisfies the unary predicate UnaryPred
template <typename UnaryPred, typename Xpr> static constexpr bool xpr_none_of() {
    return !xpr_any_of<UnaryPred, Xpr>();
};
template <typename UnaryPred, typename Xpr>
static constexpr bool xpr_none_of([[maybe_unused]] Xpr xpr, [[maybe_unused]] UnaryPred p) {
    return !xpr_any_of<std::decay_t<UnaryPred>, std::decay_t<Xpr>>();
}
  
// maximally wraps all subtrees in Xpr which not satisfies unary predicate UnaryPred with template type T
template <template <typename> typename T, typename UnaryPred, typename Xpr> constexpr auto xpr_wrap(Xpr&& xpr) {
    using Xpr_ = std::decay_t<Xpr>;
    if constexpr (!(is_scalar_field_v<Xpr_> || is_matrix_field_v<Xpr_>)) {
        return xpr;
    } else {
        if constexpr (xpr_all_of<UnaryPred, Xpr_>()) {   // maximally wrap subtrees which do not satisfy C
            return T<Xpr_>(xpr);
        } else {
            // subtree cannot be maximally wrapped, visit subtree rooted at Xpr
            if constexpr (xpr_is_binary_op<Xpr_>) {
                return typename Xpr_::template Meta<
                  decltype(xpr_wrap<T, UnaryPred>(xpr.lhs())), decltype(xpr_wrap<T, UnaryPred>(xpr.rhs()))>(
                  xpr_wrap<T, UnaryPred>(xpr.lhs()), xpr_wrap<T, UnaryPred>(xpr.rhs()));
            }
            if constexpr (xpr_is_unary_op<Xpr_>) {
                return typename Xpr_::template Meta<decltype(xpr_wrap<T, UnaryPred>(xpr.derived()))>(
                  xpr_wrap<T, UnaryPred>(xpr.derived()));
            }
            if constexpr (xpr_is_leaf<Xpr_>) {
                if constexpr (UnaryPred {}.template operator()<Xpr_>()) {
                    return T<Xpr_>(xpr);
                } else {
                    return xpr;
                }
            }
        }
    }
}
template <template <typename> typename T, typename UnaryPred, typename Xpr>
constexpr auto xpr_wrap(Xpr&& xpr, UnaryPred) {
    return xpr_wrap<T, std::decay_t<UnaryPred>>(xpr);
}

// returns the result of applying UnaryFunc(xpr, args...) to the first node in Xpr satisfying UnaryPred
template <typename UnaryFunc, typename UnaryPred, typename Xpr, typename... Args>
constexpr decltype(auto) xpr_apply_if(Xpr&& xpr, Args&&... args) {
    using Xpr_ = std::decay_t<Xpr>;
    if constexpr (UnaryPred {}.template operator()<Xpr_>()) {
        return UnaryFunc {}(xpr, std::forward<Args>(args)...);
    } else {
        if constexpr (xpr_is_binary_op<Xpr_>) {
            using LhsDerived = typename Xpr_::LhsDerived;
            using RhsDerived = typename Xpr_::RhsDerived;
            if constexpr (is_scalar_field_v<LhsDerived> || is_matrix_field_v<LhsDerived>) {
                if constexpr (xpr_any_of<UnaryPred, LhsDerived>()) {
                    return xpr_apply_if<UnaryFunc, UnaryPred>(xpr.lhs(), std::forward<Args>(args)...);
                }
            }
            if constexpr (is_scalar_field_v<RhsDerived> || is_matrix_field_v<RhsDerived>) {
                if constexpr (xpr_any_of<UnaryPred, RhsDerived>()) {
                    return xpr_apply_if<UnaryFunc, UnaryPred>(xpr.rhs(), std::forward<Args>(args)...);
                }
            }
        } else {   // unary node
            if constexpr (xpr_any_of<UnaryPred, typename Xpr_::Derived>()) {
                return xpr_apply_if<UnaryFunc, UnaryPred>(xpr.derived(), std::forward<Args>(args)...);
            }
        }
    }
}
template <typename UnaryFunc, typename UnaryPred, typename Xpr, typename... Args>
constexpr decltype(auto) xpr_apply_if(Xpr&& xpr, UnaryPred, UnaryFunc, Args&&... args) {
    return xpr_apply_if<std::decay_t<UnaryFunc>, std::decay_t<UnaryPred>>(
      std::forward<Xpr>(xpr), std::forward<Args>(args)...);
}

// apply unary functor UnaryFunc to all nodes in Xpr satisfying unary predicate UnaryPred
template <typename UnaryFunc, typename UnaryPred, typename Xpr, typename... Args>
constexpr void xpr_for_each(Xpr&& xpr, Args&&... args) {
    using Xpr_ = std::decay_t<Xpr>;
    if constexpr (UnaryPred {}.template operator()<Xpr_>()) {
        UnaryFunc {}(xpr, std::forward<Args>(args)...);
    } else {
        if constexpr (xpr_is_binary_op<Xpr_>) {
            using LhsDerived = typename Xpr_::LhsDerived;
            using RhsDerived = typename Xpr_::RhsDerived;
            if constexpr (is_scalar_field_v<LhsDerived> || is_matrix_field_v<LhsDerived>) {
                if constexpr (xpr_any_of<UnaryPred, LhsDerived>()) {
                    xpr_for_each<UnaryFunc, UnaryPred>(xpr.lhs(), std::forward<Args>(args)...);
                }
            }
            if constexpr (is_scalar_field_v<RhsDerived> || is_matrix_field_v<RhsDerived>) {
                if constexpr (xpr_any_of<UnaryPred, RhsDerived>()) {
                    xpr_for_each<UnaryFunc, UnaryPred>(xpr.rhs(), std::forward<Args>(args)...);
                }
            }
        } else {   // unary node
            if constexpr (xpr_any_of<UnaryPred, typename Xpr_::Derived>()) {
                xpr_for_each<UnaryFunc, UnaryPred>(xpr.derived(), std::forward<Args>(args)...);
            }
        }
    }
    return;
}
template <typename UnaryFunc, typename UnaryPred, typename Xpr, typename... Args>
constexpr decltype(auto) xpr_for_each(Xpr&& xpr, UnaryPred, UnaryFunc, Args&&... args) {
    return xpr_for_each<std::decay_t<UnaryFunc>, std::decay_t<UnaryPred>>(
      std::forward<Xpr>(xpr), std::forward<Args>(args)...);
}

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_XPR_HELPER_H__
