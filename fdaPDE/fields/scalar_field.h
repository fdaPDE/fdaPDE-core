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

#ifndef __SCALAR_FIELD_H__
#define __SCALAR_FIELD_H__

#include <type_traits>

#include "../utils/symbols.h"
#include "scalar_expressions.h"

namespace fdapde {
namespace core {

// a functor representing a constant field
template <int N> class ConstantField : public ScalarExpr<N, ConstantField<N>> {
    double c_;
   public:
    ConstantField(double c) : c_(c) {};
    inline double operator()(const typename static_dynamic_vector_selector<N>::type& p) const { return c_; }
};
// a functor representing a zero field
template <int N> struct ZeroField : public ConstantField<N> {
    ZeroField() : ConstantField<N>(0) {};
};

// a template class for handling general scalar fields.
template <
  int N,   // input space dimension (fdapde::Dynamic accepted)
  typename F = std::function<double(static_dynamic_vector_selector_t<N>)>>
class ScalarField : public ScalarExpr<N, ScalarField<N, F>> {
   public:
    typedef F FieldType;   // type of wrapped functor
    typedef typename static_dynamic_vector_selector<N>::type VectorType;
    typedef typename static_dynamic_matrix_selector<N, N>::type MatrixType;
    typedef ScalarExpr<N, ScalarField<N, F>> Base;
    static constexpr int DomainDimension = N;
    static_assert(
      std::is_invocable<F, VectorType>::value &&
      std::is_same<typename std::invoke_result<F, VectorType>::type, double>::value);
    // constructors
    template <int N_ = N, typename std::enable_if<N_ != Dynamic, int>::type = 0> ScalarField() {};
    template <int N_ = N, typename std::enable_if<N_ == Dynamic, int>::type = 0> ScalarField(int n) : Base(n) {};
    explicit ScalarField(const FieldType& f) : f_(f) {};

    // assignement and constructor from a ScalarExpr requires the base type F to be a std::function<>
    template <
      typename E, typename U = FieldType,
      typename std::enable_if<std::is_same<U, std::function<double(VectorType)>>::value, int>::type = 0>
    ScalarField(const ScalarExpr<N, E>& f) {
        E op = f.get();
        std::function<double(VectorType)> field_expr = [op](SVector<N> x) -> double { return op(x); };
        f_ = field_expr;
    };
    template <typename E, typename U = FieldType>
    typename std::enable_if<std::is_same<U, std::function<double(VectorType)>>::value, ScalarField<N>&>::type
    operator=(const ScalarExpr<N, E>& f) {
        E op = f.get();
        std::function<double(VectorType)> field_expr = [op](VectorType x) -> double { return op(x); };
        f_ = field_expr;
        return *this;
    };
    // assignment from lambda expression
    template <typename L, typename U = FieldType>
    typename std::enable_if<std::is_same<U, std::function<double(VectorType)>>::value, ScalarField<N>&>::type
    operator=(const L& lambda) {
        f_ = lambda;
        return *this;
    }
    // static initializers
    static ScalarField<N, ZeroField<N>> Zero() { return ScalarField<N, ZeroField<N>>(ZeroField<N>()); }
    static ScalarField<N, ConstantField<N>> Const(double c) {
        return ScalarField<N, ConstantField<N>>(ConstantField<N>(c));
    }
    // evaluation at point
    inline double operator()(const VectorType& x) const { return f_(x); };
    inline double operator()(const VectorType& x) { return f_(x); };
   protected:
    FieldType f_ {};
};

// specialization for member function pointers
template <int N, typename MemFnPtr_> struct ScalarField_MemFnBase : public ScalarExpr<N, ScalarField<N, MemFnPtr_>> {
    using MemFnPtr = fn_ptr_traits_impl<MemFnPtr_>;
    using ClassPtrType_ = std::add_pointer_t<typename MemFnPtr::ClassType>;
    using VectorType_ = typename std::tuple_element<0, typename MemFnPtr::ArgsType>::type;
    using RetType_    = typename MemFnPtr::RetType;
    using FieldType_  = typename MemFnPtr::MemFnPtrType;
    typedef typename static_dynamic_vector_selector<N>::type VectorType;
    static_assert(std::is_same<VectorType, typename std::decay<VectorType_>::type>::value);
    // constructor
    ScalarField_MemFnBase() = default;
    ScalarField_MemFnBase(ClassPtrType_ c, FieldType_ f) : c_(c), f_(f) {};
    // evaluation at point
    inline RetType_ operator()(const VectorType_& x) { return (c_->*f_)(x); };
    inline RetType_ operator()(const VectorType_& x) const { return (c_->*f_)(x); };
   protected:
    ClassPtrType_ c_ = nullptr;
    FieldType_ f_ = nullptr;
};
template <int N, typename RetType_, typename ClassType_, typename VectorType_>
struct ScalarField<N, RetType_ (ClassType_::*)(VectorType_)> :    // non-const member function pointers
    public ScalarField_MemFnBase<N, RetType_ (ClassType_::*)(VectorType_)> {
    using Base = ScalarField_MemFnBase<N, RetType_ (ClassType_::*)(VectorType_)>;
    static constexpr int DomainDimension = N;
    // constructors
    ScalarField() = default;
    ScalarField(typename Base::ClassPtrType_ c, typename Base::FieldType_ f) : Base(c, f) {};
};
template <int N, typename RetType_, typename ClassType_, typename VectorType_>
struct ScalarField<N, RetType_ (ClassType_::*)(VectorType_) const> :    // non-const member function pointers
    public ScalarField_MemFnBase<N, RetType_ (ClassType_::*)(VectorType_) const> {
    using Base = ScalarField_MemFnBase<N, RetType_ (ClassType_::*)(VectorType_) const>;
    // constructors
    ScalarField() = default;
    ScalarField(typename Base::ClassPtrType_ c, typename Base::FieldType_ f) : Base(c, f) {};
};

}   // namespace core
}   // namespace fdapde

#endif   // __SCALAR_FIELD_H__
