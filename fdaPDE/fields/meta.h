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

#ifndef __FIELD_META_H__
#define __FIELD_META_H__

namespace fdapde {

template <int Size, typename Derived> struct ScalarBase;
template <int Size, typename Derived> struct MatrixBase;
  
namespace meta {
  
// a set of template metaprogramming routines for manipulating fields expressions
template <typename T>
concept is_unary_xpr_op = requires(T t) { typename T::Derived; };
template <typename T>
concept is_binary_xpr_op = requires(T t) {
    typename T::LhsDerived;
    typename T::RhsDerived;
};
template <typename T>
concept is_coeff_xpr_op = is_binary_xpr_op<T> && requires(T t) { typename T::CoeffType; };
template <typename T>
concept is_xpr_leaf = !is_unary_xpr_op<T> && !is_binary_xpr_op<T>;

template <typename Xpr> struct is_scalar_field {
    static constexpr bool value = []() {
        if constexpr (requires(Xpr lhs) { Xpr::StaticInputSize; }) {
            return std::is_base_of_v<ScalarBase<Xpr::StaticInputSize, Xpr>, Xpr>;
        } else {
            return false;
        }
    }();
};
template <typename Xpr> static constexpr bool is_scalar_field_v = is_scalar_field<Xpr>::value;
template <typename Xpr> struct is_matrix_field {
    static constexpr bool value = []() {
        if constexpr (requires(Xpr lhs) { Xpr::StaticInputSize; }) {
            return std::is_base_of_v<MatrixBase<Xpr::StaticInputSize, Xpr>, Xpr>;
        } else {
            return false;
        }
    }();
};
template <typename Xpr> static constexpr bool is_matrix_field_v = is_matrix_field<Xpr>::value;

  
// wraps all leafs in Xpr which satisfies boolean condition C with template type T
template <template <typename> typename T, typename C, typename Xpr> constexpr auto xpr_wrap(Xpr&& xpr) {
    using Xpr_ = std::decay_t<Xpr>;
    if constexpr (!(is_scalar_field_v<Xpr_> || is_matrix_field_v<Xpr_>)) {
        return xpr;
    } else {
        if constexpr (is_binary_xpr_op<Xpr_>) {
            return
              typename Xpr_::template Meta<decltype(xpr_wrap<T, C>(xpr.lhs())), decltype(xpr_wrap<T, C>(xpr.rhs()))>(
                xpr_wrap<T, C>(xpr.lhs()), xpr_wrap<T, C>(xpr.rhs()));
        }
        if constexpr (is_unary_xpr_op<Xpr_>) {
            return typename Xpr_::template Meta<decltype(xpr_wrap<T, C>(xpr.derived()))>(xpr_wrap<T, C>(xpr.derived()));
        }
        if constexpr (is_xpr_leaf<Xpr_>) {
            if constexpr (C {}.template operator()<Xpr_>()) {
                return T<Xpr_>(xpr);
            } else {
                return xpr;
            }
        }
    }
}

// finds wheter at least one node in Xpr satisfies boolean condition C
template <typename C, typename Xpr> static constexpr bool xpr_find() {
    if constexpr (is_binary_xpr_op<Xpr>) {
        if constexpr (is_coeff_xpr_op<Xpr>) {
            if constexpr (requires(Xpr xpr) { Xpr::is_coeff_scalar_field; }) {   // MatrixCoeffWiseOp
                if constexpr (Xpr::is_coeff_scalar_field) {
                    return xpr_find<C, typename Xpr::LhsDerived>() || xpr_find<C, typename Xpr::RhsDerived>();
                }
            } else {
                if constexpr (Xpr::is_coeff_lhs) {
                    return xpr_find<C, typename Xpr::RhsDerived>();
                } else {
                    return xpr_find<C, typename Xpr::LhsDerived>();
                }
            }
        } else {
            return xpr_find<C, typename Xpr::LhsDerived>() || xpr_find<C, typename Xpr::RhsDerived>();
        }
    }
    if constexpr (is_unary_xpr_op<Xpr>) { return xpr_find<C, typename Xpr::Derived>(); }
    if constexpr (is_xpr_leaf<Xpr>) { return C {}.template operator()<Xpr>(); }
}
template <typename C, typename Xpr> static constexpr bool xpr_find(Xpr xpr) { return xpr_find<C, std::decay_t<Xpr>>(); }

// query Xpr to return the result of F on the first node in Xpr satisfying boolean condition C
template <typename F, typename C, typename Xpr, typename... Args>
constexpr decltype(auto) xpr_query(Xpr&& xpr, Args&&... args) {
    using Xpr_ = std::decay_t<Xpr>;
    if constexpr (C {}.template operator()<Xpr_>()) {
        return F {}(xpr, std::forward<Args>(args)...);
    } else {
        if constexpr (is_binary_xpr_op<Xpr_>) {
            if constexpr (xpr_find<C, typename Xpr_::LhsDerived>()) {
                return xpr_query<F, C>(xpr.lhs(), std::forward<Args>(args)...);
            }
            if constexpr (xpr_find<C, typename Xpr_::RhsDerived>()) {
                return xpr_query<F, C>(xpr.rhs(), std::forward<Args>(args)...);
            }
        } else if constexpr (xpr_find<C, typename Xpr_::Derived>()) {
            return xpr_query<F, C>(xpr.derived(), std::forward<Args>(args)...);
        }
    }
}

// apply functor F to all nodes in Xpr satisfying condition C
template <typename F, typename C, typename Xpr, typename... Args>
constexpr void xpr_apply_if(Xpr&& xpr, Args&&... args) {
    using Xpr_ = std::decay_t<Xpr>;
    if constexpr (C {}.template operator()<Xpr_>()) {
        F {}(xpr, std::forward<Args>(args)...);
    } else {
        if constexpr (is_binary_xpr_op<Xpr_>) {
            if constexpr (xpr_find<C, typename Xpr_::LhsDerived>()) {
                xpr_apply_if<F, C>(xpr.lhs(), std::forward<Args>(args)...);
            }
            if constexpr (xpr_find<C, typename Xpr_::RhsDerived>()) {
                xpr_apply_if<F, C>(xpr.rhs(), std::forward<Args>(args)...);
            }
        } else if constexpr (xpr_find<C, typename Xpr_::Derived>()) {
            xpr_apply_if<F, C>(xpr.derived(), std::forward<Args>(args)...);
        }
    }
    return;
}

}   // namespace meta
}   // namespace fdapde

#endif   // __FIELD_META_H__
