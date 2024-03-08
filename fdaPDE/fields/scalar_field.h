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
    using VectorType = typename static_dynamic_vector_selector<N>::type;
    double c_ = 0;
   public:
    explicit ConstantField(double c) : c_(c) { }
    inline double operator()([[maybe_unused]] const VectorType& p) const { return c_; }
};
// a functor representing a zero field
template <int N> struct ZeroField : public ConstantField<N> {
    explicit ZeroField() : ConstantField<N>(0) { }
};

// a template class for handling general scalar fields.
template <
  int N,   // input space dimension (fdapde::Dynamic accepted)
  typename F = std::function<double(static_dynamic_vector_selector_t<N>)>>
class ScalarField : public ScalarExpr<N, ScalarField<N, F>> {
   public:
    using FieldType = F;   // type of wrapped functor
    using VectorType = typename static_dynamic_vector_selector<N>::type;
    using MatrixType = typename static_dynamic_matrix_selector<N, N>::type;
    using Base = ScalarExpr<N, ScalarField<N, F>>;
    static constexpr int DomainDimension = N;
    static_assert(
      std::is_invocable<F, VectorType>::value &&
      std::is_same<typename std::invoke_result<F, VectorType>::type, double>::value);
    // constructors
    explicit ScalarField() requires(N != Dynamic) { }
    explicit ScalarField(int n) requires (N == Dynamic) : Base(n) { }
    explicit ScalarField(const FieldType& f) : f_(f) {};
    // assignement and constructor from a ScalarExpr requires the base type F to be a std::function<>
    template <typename E>
    ScalarField(const ScalarExpr<N, E>& f)
        requires(std::is_same<FieldType, std::function<double(VectorType)>>::value) {
        E op = f.get();
	f_ = [op](SVector<N> x) -> double { return op(x); };
    }
    template <typename E>
    ScalarField& operator=(const ScalarExpr<N, E>& f)
        requires(std::is_same<FieldType, std::function<double(VectorType)>>::value) {
        E op = f.get();
        f_ = [op](VectorType x) -> double { return op(x); };
        return *this;
    }
    // assignment from lambda expression
    template <typename L>
    ScalarField& operator=(const L& lambda)
        requires(std::is_same<FieldType, std::function<double(VectorType)>>::value) {
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
