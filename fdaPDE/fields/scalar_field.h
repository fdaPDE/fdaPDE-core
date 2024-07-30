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

template <int Size, typename Derived> struct ScalarBase;

// unary negation operation
template <typename Derived>
class ScalarUnaryMinus : public ScalarBase<Derived::StaticInnerSize, ScalarUnaryMinus<Derived>> {
   private:
    using Base = ScalarBase<Derived::StaticInnerSize, ScalarUnaryMinus<Derived>>;
    typename internals::ref_select<const Derived>::type derived_;
   public:
    using VectorType = typename Derived::VectorType;
    static constexpr int StaticInnerSize = Derived::StaticInnerSize;
    static constexpr int NestAsRef = 0;
    ScalarUnaryMinus(const Derived& derived) requires(StaticInnerSize == Dynamic)
        : Base(derived.inner_size()), derived_(derived) { }
    explicit constexpr ScalarUnaryMinus(const Derived& derived) requires(StaticInnerSize != Dynamic)
        : Base(), derived_(derived) { }
    constexpr double operator()(const VectorType& p) const { return -derived_(p); }
    template <typename... Args> ScalarUnaryMinus<Derived>& forward(Args&&... args) {
        derived_.forward(std::forward<Args>(args)...);
        return *this;
    }
    constexpr int static_inner_size() const { return StaticInnerSize; }
};

template <typename Derived, typename UnaryFunctor>
class ScalarUnaryOp : public ScalarBase<Derived::StaticInnerSize, ScalarUnaryOp<Derived, UnaryFunctor>> {
   private:
    using Base = ScalarBase<Derived::StaticInnerSize, ScalarUnaryOp<Derived, UnaryFunctor>>;
    typename internals::ref_select<const Derived>::type derived_;
    UnaryFunctor op_;
   public:
    using VectorType = typename Derived::VectorType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInnerSize = Derived::StaticInnerSize;
    static constexpr int NestAsRef = 0;
    ScalarUnaryOp(const Derived& derived) requires(StaticInnerSize == Dynamic)
        : Base(derived.inner_size()), derived_(derived) { }
    explicit constexpr ScalarUnaryOp(const Derived& derived) requires(StaticInnerSize != Dynamic)
        : Base(), derived_(derived) { }
    constexpr ScalarUnaryOp(const Derived& derived, const UnaryFunctor& op) requires(StaticInnerSize != Dynamic)
        : Base(), derived_(derived), op_(op) { }
    constexpr int static_inner_size() const { return StaticInnerSize; }
    constexpr Scalar operator()(const VectorType& p) const {
        if constexpr (StaticInnerSize == Dynamic) { fdapde_assert(p.rows() == Base::inner_size()); }
        return op_(derived_(p));
    }
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

template <typename Lhs, typename Rhs, typename BinaryOperation>
class ScalarBinOp : public ScalarBase<Lhs::StaticInnerSize, ScalarBinOp<Lhs, Rhs, BinaryOperation>> {
    fdapde_static_assert(
      Lhs::StaticInnerSize == Rhs::StaticInnerSize, YOU_MIXED_SCALAR_FUNCTIONS_WITH_DIFFERENT_STATIC_INNER_SIZE);
    fdapde_static_assert(
      std::is_same_v<typename Lhs::VectorType FDAPDE_COMMA typename Rhs::VectorType>,
      YOU_MIXED_SCALAR_FUNCTIONS_WITH_DIFFERENT_VECTOR_TYPES);
    fdapde_static_assert(
      std::is_same_v<typename Lhs::Scalar FDAPDE_COMMA typename Rhs::Scalar>,
      YOU_MIXED_SCALAR_FUNCTIONS_WITH_DIFFERENT_SCALAR_TYPES);
   private:
    using Base = ScalarBase<Lhs::StaticInnerSize, ScalarBinOp<Lhs, Rhs, BinaryOperation>>;
    typename internals::ref_select<const Lhs>::type lhs_;
    typename internals::ref_select<const Rhs>::type rhs_;
    BinaryOperation op_;
   public:
    using VectorType = typename Lhs::VectorType;
    using Scalar = typename Lhs::Scalar;
    static constexpr int StaticInnerSize = Lhs::StaticInnerSize;
    static constexpr int NestAsRef = 0;
    // constructor
    ScalarBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) requires(StaticInnerSize == Dynamic) :
        Base(lhs.inner_size()), lhs_(lhs), rhs_(rhs), op_(op) {
        fdapde_assert(lhs.inner_size() == rhs.inner_size());
    }
    constexpr ScalarBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) requires(StaticInnerSize != Dynamic) :
        Base(), lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr double operator()(const VectorType& p) const {
        if constexpr (StaticInnerSize == Dynamic) { fdapde_assert(p.rows() == Base::inner_size()); }
        return op_(lhs_(p), rhs_(p));
    }
    constexpr int static_inner_size() const { return StaticInnerSize; }
    template <typename... Args> constexpr ScalarBinOp<Lhs, Rhs, BinaryOperation>& forward(Args&&... args) {
        lhs_.forward(std::forward<Args>(args)...);
        rhs_.forward(std::forward<Args>(args)...);
        return *this;
    }
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

template <typename Lhs, typename Rhs, typename BinaryOperation>
struct ScalarCoeffOp :
    public ScalarBase<
      std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>::StaticInnerSize,
      ScalarCoeffOp<Lhs, Rhs, BinaryOperation>> {
    using CoeffType = std::conditional_t<std::is_arithmetic_v<Lhs>, Lhs, Rhs>;
    using Derived = std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>;
    fdapde_static_assert(
      std::is_convertible_v<CoeffType FDAPDE_COMMA typename Derived::Scalar> && std::is_arithmetic_v<CoeffType>,
      COEFFICIENT_IN_BINARY_OPERATION_NOT_CONVERTIBLE_TO_SCALAR_TYPE);
   private:
    using Base = ScalarBase<Derived::StaticInnerSize, ScalarCoeffOp<Lhs, Rhs, BinaryOperation>>;
    typename internals::ref_select<const Derived>::type derived_;
    BinaryOperation op_;
    CoeffType coeff_;
    static constexpr bool is_coeff_lhs =
      std::is_arithmetic_v<Lhs>;   // whether to perform op_(derived_(p), coeff) or op_(coeff, derived_(p))
   public:
    using VectorType = typename Derived::VectorType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInnerSize = Derived::StaticInnerSize;
    static constexpr int NestAsRef = 0;
    // constructor
    ScalarCoeffOp(const Derived& derived, CoeffType coeff, BinaryOperation op) requires(StaticInnerSize == Dynamic) :
        Base(derived.inner_size()), derived_(derived), coeff_(coeff), op_(op) { }
    constexpr ScalarCoeffOp(const Derived& derived, CoeffType coeff, BinaryOperation op)
        requires(StaticInnerSize != Dynamic)
        : Base(), derived_(derived), coeff_(coeff), op_(op) { }
    constexpr double operator()(const VectorType& p) const {
        if constexpr (StaticInnerSize == Dynamic) { fdapde_assert(p.rows() == Base::inner_size()); }
        if constexpr (is_coeff_lhs) {
            return op_(coeff_, derived_(p));
        } else {
            return op_(derived_(p), coeff_);
        }
    }
    template <typename... Args> constexpr ScalarCoeffOp<Lhs, Rhs, BinaryOperation>& forward(Args&&... args) {
        derived_.forward(std::forward<Args>(args)...);
        return *this;
    }
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
        return ScalarCoeffOp<Coeff, Derived, FUNCTOR> {rhs.derived(), lhs, FUNCTOR()};                                 \
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
    using Base = ScalarBase<Size, ScalarField<Size, FunctorType>>;
    using traits = fn_ptr_traits<&FunctorType::operator()>;
    fdapde_static_assert(traits::n_args == 1, PROVIDED_FUNCTOR_MUST_ACCEPT_ONLY_ONE_ARGUMENT);
   public:
    static constexpr int StaticInnerSize = Size;      // dimensionality of base space (can be Dynamic)
    static constexpr int NestAsRef = 0;               // whether to store the node by reference of by copy
    using VectorType = std::tuple_element_t<0, typename traits::ArgsType>;
    using Scalar = typename std::invoke_result<FunctorType, VectorType>::type;
    // constructors
    constexpr ScalarField() requires(Size != Dynamic) : f_() { }
    explicit ScalarField(int n) requires(Size == Dynamic) : Base(n) { }
    constexpr explicit ScalarField(const FunctorType& f) : f_(f) {};
    template <typename Expr>
    ScalarField(const ScalarBase<Size, Expr>& f) {
        fdapde_static_assert(
          std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(VectorType)>> &&
            std::is_convertible_v<typename Expr::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_FUNCTION_ASSIGNMENT);
        Expr expr = f.get();
        f_ = [expr](const VectorType& x) { return expr(x); };
    }
    template <typename Expr>
    ScalarField& operator=(const ScalarBase<Size, Expr>& f) {
        fdapde_static_assert(
          std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(VectorType)>> &&
            std::is_convertible_v<typename Expr::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_FUNCTION_ASSIGNMENT);
        Expr expr = f.get();
        f_ = [expr](const VectorType& x) { return expr(x); };
        return *this;
    }
    // assignment from lambda expression
    template <typename LamdaType>
    ScalarField& operator=(const LamdaType& lambda) {
        fdapde_static_assert(
          std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(VectorType)>> &&
            std::is_convertible_v<
              typename std::invoke_result<LamdaType FDAPDE_COMMA VectorType>::type FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_FUNCTION_ASSIGNMENT);
        f_ = lambda;
        return *this;
    }
    // static initializers
    struct ConstantFunction : public ScalarBase<Size, ConstantFunction> {
        Scalar c_ = 0;
        ConstantFunction(Scalar c) : c_(c) { }
        constexpr Scalar operator()([[maybe_unused]] const VectorType& x) const { return c_; }
    };
    static constexpr ConstantFunction Constant(Scalar c) { return ConstantFunction(c); }
    static constexpr ConstantFunction Zero() { return ConstantFunction(0.0); }
    constexpr int static_inner_size() const { return StaticInnerSize; }
    // evaluation at point
    constexpr double operator()(const VectorType& x) const { return f_(x); }
    constexpr double operator()(const VectorType& x) { return f_(x); }
    template <typename... Args> constexpr ScalarField<Size, FunctorType_>& forward(Args&&... args) {
        if constexpr (std::is_invocable_v<FunctorType_, Args&&...>) f_(std::forward<Args>(args)...);
        return *this;
    }
   protected:
    FunctorType f_ {};
};

template <typename Derived, int Order> struct PartialDerivative;

template <typename Derived>
struct PartialDerivative<Derived, 1> : public ScalarBase<Derived::StaticInnerSize, PartialDerivative<Derived, 1>> {
   private:
    using Base = ScalarBase<Derived::StaticInnerSize, PartialDerivative<Derived, 1>>;
    typename internals::ref_select<const Derived>::type f_;
    int i_;
    double step_;
   public:
    using VectorType = std::decay_t<typename Derived::VectorType>;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInnerSize = Derived::StaticInnerSize;
    static constexpr int NestAsRef = 0;
    constexpr PartialDerivative(const Derived& f, int i, double step) requires(StaticInnerSize != Dynamic)
        : Base(), f_(f), i_(i), step_(step) { }
    constexpr PartialDerivative(const Derived& f, int i, double step) requires(StaticInnerSize == Dynamic)
        : Base(f.inner_size()), f_(f), i_(i), step_(step) { }
    constexpr Scalar operator()(VectorType x) const {
        if constexpr (std::is_arithmetic<VectorType>::value) {
            return (f_(x + step_) - f_(x - step_)) / (2 * step_);
        } else {
            Scalar res = 0;
            x[i_] = x[i_] + step_;
            res = f_(x);
            x[i_] = x[i_] - 2 * step_;
            return (res - f_(x)) / (2 * step_);
        }
    }
};

template <typename Derived>
struct PartialDerivative<Derived, 2> : public ScalarBase<Derived::StaticInnerSize, PartialDerivative<Derived, 2>> {
   private:
    using Base = ScalarBase<Derived::StaticInnerSize, PartialDerivative<Derived, 2>>;
    typename internals::ref_select<const Derived>::type f_;
    int i_, j_;
    double h_;
   public:
    using VectorType = std::decay_t<typename Derived::VectorType>;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInnerSize = Derived::StaticInnerSize;
    static constexpr int NestAsRef = 0;
    constexpr PartialDerivative(const Derived& f, int i, int j, double h) requires(StaticInnerSize != Dynamic)
        : Base(), f_(f), i_(i), h_(h) { }
    constexpr PartialDerivative(const Derived& f, int i, int j, double h) requires(StaticInnerSize == Dynamic)
        : Base(f.inner_size()), f_(f), i_(i), h_(h) { }
    constexpr Scalar operator()(VectorType x) const {
        if constexpr (std::is_arithmetic<VectorType>::value) {
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
};

// Base class for scalar field expressions
template <int Size, typename Derived_> struct ScalarBase {
    using Derived = Derived_;
    constexpr explicit ScalarBase() {
        fdapde_static_assert(Size != Dynamic, THIS_METHOD_IS_ONLY_FOR_STATIC_SIZED_FUNCTIONS);
    }
    explicit ScalarBase(int dynamic_inner_size) : dynamic_inner_size_(dynamic_inner_size) {
        fdapde_static_assert(Size == Dynamic, THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_FUNCTIONS);
    }
    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    // forward args... to all nodes of the expression. This method does nothing if not redefined in Derived
    template <typename... Args> constexpr void forward([[maybe_unused]] Args&&... args) const { return; }
    // unary minus operator
    constexpr ScalarUnaryMinus<Derived> operator-() const { return ScalarUnaryMinus<Derived>(derived()); }
    constexpr int inner_size() const { return (Size == Dynamic) ? dynamic_inner_size_ : derived().static_inner_size(); }
    void resize(int dynamic_inner_size) {
        fdapde_static_assert(Size == Dynamic, YOU_CALLED_A_DYNAMIC_METHOD_ON_A_STATIC_SIZED_FUNCTION);
        dynamic_inner_size_ = dynamic_inner_size;
    }
    constexpr void set_step(double step) { step_ = step; }
    constexpr double step() const { return step_; }

    constexpr PartialDerivative<Derived, 1> partial(int i) const requires(Size > 1 || Size == Dynamic) {
        return PartialDerivative<Derived, 1>(derived(), i, step_);
    }
    constexpr PartialDerivative<Derived, 1> partial() const requires(Size == 1){ return {derived(), 0, step_}; }
    constexpr PartialDerivative<Derived, 2> partial(int i, int j) const requires(Size > 1 || Size == Dynamic) {
        return PartialDerivative<Derived, 2>(derived(), i, j, step_);
    }
   protected:
    int dynamic_inner_size_ = 0;   // run-time base space dimension
    double step_ = 1e-3;           // step size used in derivative approximation
};
  
}   // namespace fdapde

#endif   // __SCALAR_FIELD_H__
