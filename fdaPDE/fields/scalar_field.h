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

namespace fdapde {

template <int StaticInputSize, typename Derived> struct ScalarBase;
  
template <typename Derived_, typename UnaryFunctor>
struct ScalarUnaryOp : public ScalarBase<Derived_::StaticInputSize, ScalarUnaryOp<Derived_, UnaryFunctor>> {
    using Derived = Derived_;
    using UnaryOp = UnaryFunctor;
    using Base = ScalarBase<Derived::StaticInputSize, ScalarUnaryOp<Derived, UnaryFunctor>>;
    using InputType = typename Derived::InputType;
    using Scalar = decltype(std::declval<UnaryFunctor>().operator()(std::declval<typename Derived::Scalar>()));
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;

    constexpr ScalarUnaryOp(const Derived_& derived, const UnaryFunctor& op) : Base(), derived_(derived), op_(op) { }
    constexpr ScalarUnaryOp(const Derived_& derived) : ScalarUnaryOp(derived, UnaryFunctor()) { }
    constexpr Scalar operator()(const InputType& p) const {
        if constexpr (StaticInputSize == Dynamic) { fdapde_assert(p.rows() == Base::input_size()); }
        return op_(derived_(p));
    }
    constexpr int input_size() const { return derived_.input_size(); }
    constexpr const Derived& derived() const { return derived_; }
   private:
    typename internals::ref_select<const Derived>::type derived_;
    UnaryFunctor op_;
};
template <int Size, typename Derived> constexpr auto sin(const ScalarBase<Size, Derived>& f) {
    return ScalarUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::sin(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto cos(const ScalarBase<Size, Derived>& f) {
    return ScalarUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::cos(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto logn(const ScalarBase<Size, Derived>& f) {
    return ScalarUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::log(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto exp(const ScalarBase<Size, Derived>& f) {
    return ScalarUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::exp(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto abs(const ScalarBase<Size, Derived>& f) {
    return ScalarUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::abs(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto sqrt(const ScalarBase<Size, Derived>& f) {
    return ScalarUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::sqrt(x); })>(f.derived());
}

namespace internals {
class pow_t {
    int i_ = 0;
   public:
    constexpr explicit pow_t(int i) : i_(i) { }
    template <typename T> constexpr T operator()(T&& t) const { return std::pow(t, i_); }
};
}   // namespace internals

template <int Size, typename Derived> constexpr auto pow(const ScalarBase<Size, Derived>& f, int i) {
    return ScalarUnaryOp<Derived, internals::pow_t>(f.derived(), internals::pow_t(i));
}

template <typename Lhs_, typename Rhs_, typename BinaryOperation>
class ScalarBinOp : public ScalarBase<Lhs_::StaticInputSize, ScalarBinOp<Lhs_, Rhs_, BinaryOperation>> {
    fdapde_static_assert(
      Lhs_::StaticInputSize == Rhs_::StaticInputSize, YOU_MIXED_SCALAR_FUNCTIONS_WITH_DIFFERENT_STATIC_INNER_SIZES);
    fdapde_static_assert(
      std::is_convertible_v<typename Lhs_::Scalar FDAPDE_COMMA typename Rhs_::Scalar>,
      YOU_MIXED_SCALAR_FIELDS_WITH_NON_CONVERTIBLE_SCALAR_OUTPUT_TYPES);
   public:
    using LhsDerived = Lhs_;
    using RhsDerived = Rhs_;
    using BinaryOp = BinaryOperation;
    using Base = ScalarBase<LhsDerived::StaticInputSize, ScalarBinOp<LhsDerived, RhsDerived, BinaryOperation>>;
    using InputType = typename LhsDerived::InputType;
    using Scalar = typename LhsDerived::Scalar;
    static constexpr int StaticInputSize = LhsDerived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = LhsDerived::XprBits | RhsDerived::XprBits;

    ScalarBinOp(const Lhs_& lhs, const Rhs_& rhs, BinaryOperation op) requires(StaticInputSize == Dynamic) :
        Base(), lhs_(lhs), rhs_(rhs), op_(op) {
        fdapde_assert(lhs.input_size() == rhs.input_size());
    }
    constexpr ScalarBinOp(const Lhs_& lhs, const Rhs_& rhs, BinaryOperation op) requires(StaticInputSize != Dynamic) :
        Base(), lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr Scalar operator()(const InputType& p) const {
        fdapde_static_assert(
          std::is_same_v<typename Lhs_::InputType FDAPDE_COMMA typename Rhs_::InputType>,
          YOU_MIXED_SCALAR_FIELDS_WITH_DIFFERENT_INPUT_VECTOR_TYPES);
        if constexpr (StaticInputSize == Dynamic) { fdapde_assert(p.rows() == Base::input_size()); }
        return op_(lhs_(p), rhs_(p));
    }
    constexpr int input_size() const { return lhs_.input_size(); }
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
    constexpr LhsDerived& lhs() { return lhs_; }
    constexpr RhsDerived& rhs() { return rhs_; }
   private:
    typename internals::ref_select<LhsDerived>::type lhs_;
    typename internals::ref_select<RhsDerived>::type rhs_;
    BinaryOperation op_;
};

template <int Size, typename Lhs, typename Rhs>
constexpr ScalarBinOp<Lhs, Rhs, std::plus<>>
operator+(const ScalarBase<Size, Lhs>& lhs, const ScalarBase<Size, Rhs>& rhs) {
    return ScalarBinOp<Lhs, Rhs, std::plus<>> {lhs.derived(), rhs.derived(), std::plus<>()};
}
template <int Size, typename Lhs, typename Rhs>
constexpr ScalarBinOp<Lhs, Rhs, std::minus<>>
operator-(const ScalarBase<Size, Lhs>& lhs, const ScalarBase<Size, Rhs>& rhs) {
    return ScalarBinOp<Lhs, Rhs, std::minus<>> {lhs.derived(), rhs.derived(), std::minus<>()};
}
template <int Size, typename Lhs, typename Rhs>
constexpr ScalarBinOp<Lhs, Rhs, std::multiplies<>>
operator*(const ScalarBase<Size, Lhs>& lhs, const ScalarBase<Size, Rhs>& rhs) {
    return ScalarBinOp<Lhs, Rhs, std::multiplies<>> {lhs.derived(), rhs.derived(), std::multiplies<>()};
}
template <int Size, typename Lhs, typename Rhs>
constexpr ScalarBinOp<Lhs, Rhs, std::divides<>>
operator/(const ScalarBase<Size, Lhs>& lhs, const ScalarBase<Size, Rhs>& rhs) {
    return ScalarBinOp<Lhs, Rhs, std::divides<>> {lhs.derived(), rhs.derived(), std::divides<>()};
}

template <typename Lhs_, typename Rhs_, typename BinaryOperation>
struct ScalarCoeffOp :
    public ScalarBase<
      std::conditional_t<std::is_arithmetic_v<Lhs_>, Rhs_, Lhs_>::StaticInputSize,
      ScalarCoeffOp<Lhs_, Rhs_, BinaryOperation>> {
   private:
    // keep this private to avoid to consider ScalarCoeffOp as a unary node
    using Derived = std::conditional_t<std::is_arithmetic_v<Lhs_>, Rhs_, Lhs_>;
   public:
    using CoeffType = std::conditional_t<std::is_arithmetic_v<Lhs_>, Lhs_, Rhs_>;
    fdapde_static_assert(
      std::is_convertible_v<CoeffType FDAPDE_COMMA typename Derived::Scalar> && std::is_arithmetic_v<CoeffType>,
      COEFFICIENT_IN_BINARY_OPERATION_NOT_CONVERTIBLE_TO_SCALAR_TYPE);
    using LhsDerived = Lhs_;
    using RhsDerived = Rhs_;
    using BinaryOp = BinaryOperation;
    static constexpr bool is_coeff_lhs =
      std::is_arithmetic_v<Lhs_>;   // whether to perform op_(xpr_(p), coeff) or op_(coeff, xpr_(p))
    using Base = ScalarBase<Derived::StaticInputSize, ScalarCoeffOp<LhsDerived, RhsDerived, BinaryOperation>>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;

    constexpr ScalarCoeffOp(const Lhs_& lhs, const Rhs_& rhs, BinaryOperation op) :
        Base(), lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr Scalar operator()(const InputType& p) const {
        if constexpr (StaticInputSize == Dynamic) { fdapde_assert(p.rows() == Base::input_size()); }
        if constexpr (is_coeff_lhs) {
            return op_(Scalar(lhs_), rhs_(p));
        } else {
            return op_(lhs_(p), Scalar(rhs_));
        }
    }
    constexpr int input_size() const {
        if constexpr (is_coeff_lhs) {
            return lhs_.input_size();
        } else {
            return rhs_.input_size();
        }
    }
    constexpr LhsDerived& lhs() { return lhs_; }
    constexpr RhsDerived& rhs() { return rhs_; }
   private:
    typename internals::ref_select<LhsDerived>::type lhs_;
    typename internals::ref_select<RhsDerived>::type rhs_;
    BinaryOperation op_;
};
  
#define FDAPDE_DEFINE_SCALAR_COEFF_OP(OPERATOR, FUNCTOR)                                                               \
    template <int Size, typename Derived, typename Coeff>                                                              \
    constexpr ScalarCoeffOp<Derived, Coeff, FUNCTOR> OPERATOR(const ScalarBase<Size, Derived>& lhs, Coeff rhs)         \
        requires(std::is_arithmetic_v<Coeff>) {                                                                        \
        return ScalarCoeffOp<Derived, Coeff, FUNCTOR> {lhs.derived(), rhs, FUNCTOR()};                                 \
    }                                                                                                                  \
    template <int Size, typename Derived, typename Coeff>                                                              \
    constexpr ScalarCoeffOp<Coeff, Derived, FUNCTOR> OPERATOR(Coeff lhs, const ScalarBase<Size, Derived>& rhs)         \
        requires(std::is_arithmetic_v<Coeff>) {                                                                        \
        return ScalarCoeffOp<Coeff, Derived, FUNCTOR> {lhs, rhs.derived(), FUNCTOR()};                                 \
    }

FDAPDE_DEFINE_SCALAR_COEFF_OP(operator+, std::plus<>      )
FDAPDE_DEFINE_SCALAR_COEFF_OP(operator-, std::minus<>     )
FDAPDE_DEFINE_SCALAR_COEFF_OP(operator*, std::multiplies<>)
FDAPDE_DEFINE_SCALAR_COEFF_OP(operator/, std::divides<>   )

  template <
    int Size,   // input space dimension (fdapde::Dynamic accepted)
    typename FunctorType_ = std::function<double(static_dynamic_vector_selector_t<Size>)>>
  class ScalarField : public ScalarBase<Size, ScalarField<Size, FunctorType_>> {
    using FunctorType = std::decay_t<FunctorType_>;   // type of wrapped functor
    using traits = fn_ptr_traits<&FunctorType::operator()>;
    fdapde_static_assert(traits::n_args == 1, PROVIDED_FUNCTOR_MUST_ACCEPT_ONLY_ONE_ARGUMENT);
   public:
    using Base = ScalarBase<Size, ScalarField<Size, FunctorType>>;
    using InputType = std::tuple_element_t<0, typename traits::ArgsType>;
    using Scalar = typename std::invoke_result<FunctorType, InputType>::type;
    static constexpr int StaticInputSize = Size;      // dimensionality of base space (can be Dynamic)
    static constexpr int NestAsRef = 0;               // whether to store the node by reference of by copy
    static constexpr int XprBits = 0;                 // bits which carries implementation specific informations

    constexpr ScalarField() requires(StaticInputSize != Dynamic) : f_() { }
    explicit ScalarField(int n) requires(StaticInputSize == Dynamic)
        : Base(), f_() { }
    constexpr explicit ScalarField(const FunctorType& f) : f_(f) {};
    template <typename Expr>
    ScalarField(const ScalarBase<Size, Expr>& f) {
        fdapde_static_assert(
          std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(InputType)>> &&
            std::is_convertible_v<typename Expr::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_FUNCTION_ASSIGNMENT);
        Expr expr = f.get();
        f_ = [expr](const InputType& x) { return expr(x); };
    }
    template <typename Expr>
    ScalarField& operator=(const ScalarBase<Size, Expr>& f) {
        fdapde_static_assert(
          std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(InputType)>> &&
            std::is_convertible_v<typename Expr::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_FUNCTION_ASSIGNMENT);
        Expr expr = f.get();
        f_ = [expr](const InputType& x) { return expr(x); };
        return *this;
    }
    // assignment from lambda expression
    template <typename LamdaType>
    ScalarField& operator=(const LamdaType& lambda) {
        fdapde_static_assert(
          std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(InputType)>> &&
            std::is_convertible_v<
              typename std::invoke_result<LamdaType FDAPDE_COMMA InputType>::type FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_FUNCTION_ASSIGNMENT);
        f_ = lambda;
        return *this;
    }
    // static initializers
    struct ConstantFunction : public ScalarBase<Size, ConstantFunction> {
        Scalar c_ = 0;
        ConstantFunction(Scalar c) : c_(c) { }
        constexpr Scalar operator()([[maybe_unused]] const InputType& x) const { return c_; }
    };
    static constexpr ConstantFunction Constant(Scalar c) { return ConstantFunction(c); }
    static constexpr ConstantFunction Zero() { return ConstantFunction(0.0); }
    constexpr int input_size() const { return StaticInputSize == Dynamic ? dynamic_input_size_ : StaticInputSize; }
    // evaluation at point
    constexpr Scalar operator()(const InputType& x) const { return f_(x); }
    constexpr Scalar operator()(const InputType& x) { return f_(x); }
    void resize(int dynamic_input_size) {
        fdapde_static_assert(StaticInputSize == Dynamic, YOU_CALLED_A_DYNAMIC_METHOD_ON_A_STATIC_SIZED_FIELD);
        dynamic_input_size_ = dynamic_input_size;
    }
   protected:
    int dynamic_input_size_ = 0;   // run-time base space dimension
    FunctorType f_ {};
};

template <typename Derived, int Order> struct PartialDerivative;

template <typename Derived_>
struct PartialDerivative<Derived_, 1> : public ScalarBase<Derived_::StaticInputSize, PartialDerivative<Derived_, 1>> {
    using Derived = Derived_;
    using Base = ScalarBase<Derived::StaticInputSize, PartialDerivative<Derived, 1>>;
    using InputType = std::decay_t<typename Derived::InputType>;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;

    constexpr PartialDerivative() = default;
    constexpr PartialDerivative(const Derived_& f, int i) : Base(), f_(f), i_(i) { }
    constexpr PartialDerivative(const Derived_& f, int i, double h) : Base(), f_(f), i_(i), h_(h) { }
    constexpr Scalar operator()(InputType x) const {
        if constexpr (std::is_arithmetic<InputType>::value) {
            return (f_(x + h_) - f_(x - h_)) / (2 * h_);
        } else {
            Scalar res = 0;
            x[i_] = x[i_] + h_;
            res = f_(x);
            x[i_] = x[i_] - 2 * h_;
            return (res - f_(x)) / (2 * h_);
        }
    }
    constexpr int input_size() const { return f_.input_size(); }
    constexpr const Derived& derived() const { return f_; }
   private:
    typename internals::ref_select<Derived>::type f_;
    int i_;
    double h_ = 1e-3;
};

template <typename Derived_>
struct PartialDerivative<Derived_, 2> : public ScalarBase<Derived_::StaticInputSize, PartialDerivative<Derived_, 2>> {
    using Derived = Derived_;
    using Base = ScalarBase<Derived::StaticInputSize, PartialDerivative<Derived, 2>>;
    using InputType = std::decay_t<typename Derived::InputType>;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;

    constexpr PartialDerivative() = default;
    constexpr PartialDerivative(const Derived& f, int i, int j) : Base(), f_(f), i_(i), j_(j) { }
    constexpr PartialDerivative(const Derived& f, int i, int j, double h) : Base(), f_(f), i_(i), j_(j), h_(h) { }
    constexpr Scalar operator()(InputType x) const {
        if constexpr (std::is_arithmetic<InputType>::value) {
            return i_ != j_ ?
                     ((f_(x + 2 * h_) - 2 * f_(x) + f_(x - 2 * h_)) / (4 * h_ * h_)) :
                     ((-f_(x + 2 * h_) + 16 * f_(x + h_) - 30 * f_(x) + 16 * f_(x - h_) - derived(x - 2 * h_)) /
                      (12 * h_ * h_));
        } else {
            Scalar res = 0;
            if (i_ != j_) {
                // (f(x + h_i + h_j) - f(x + h_i - h_j) - f(x - h_i + h_j) + f(x - h_i - h_j)) / (4 * h^2)
                x[i_] = x[i_] + h_; x[j_] = x[j_] + h_;
                res = f_(x);
                x[j_] = x[j_] - 2 * h_;
                res = res - f_(x);
                x[i_] = x[i_] - 2 * h_;
                res = res + f_(x);
                x[j_] = x[j_] + 2 * h_;
                return (res - f_(x)) / (4 * h_ * h_);
            } else {
                // (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x) + 16 * f(x - h) - f(x - 2 * h)) / (12 * h^2)
                x[i_] = x[i_] + 2 * h_;
                res = -f_(x);
                x[i_] = x[i_] - h_;
                res = res + 16 * f_(x);
                x[i_] = x[i_] - h_;
                res = res - 30 * f_(x);
                x[i_] = x[i_] - h_;
                res = res + 16 * f_(x);
                x[i_] = x[i_] - h_;
                return (res - f_(x)) / (12 * h_ * h_);
            }
        }
    }
    constexpr int input_size() const { return f_.input_size(); }
    constexpr const Derived& derived() const { return f_; }
   private:
    typename internals::ref_select<Derived>::type f_;
    int i_, j_;
    double h_ = 1e-3;
};

template <int Size, typename Derived> struct ScalarBase {
    constexpr ScalarBase() = default;
  
    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    constexpr auto operator-() const {   // unary minus
        return ScalarUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return -x; })>(derived());
    }
    // differential quantities
    constexpr void set_step(double step) { step_ = step; }
    constexpr double step() const { return step_; }
    constexpr PartialDerivative<Derived, 1> partial(int i) const requires(Size > 1 || Size == Dynamic) {
        return PartialDerivative<Derived, 1>(derived(), i, step_);
    }
    constexpr PartialDerivative<Derived, 2> partial(int i, int j) const requires(Size > 1 || Size == Dynamic) {
        return PartialDerivative<Derived, 2>(derived(), i, j, step_);
    }
    constexpr auto gradient() const { return Gradient<Derived>(derived()); }
    constexpr auto hessian() const { return Hessian<Derived>(derived()); }
    constexpr auto laplacian() const { return Laplacian<Derived>(derived()); }
   protected:
    double step_ = 1e-3;   // step size used in derivative approximation
};

}   // namespace fdapde

#endif   // __SCALAR_FIELD_H__
