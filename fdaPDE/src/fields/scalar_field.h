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

#ifndef __FDAPDE_SCALAR_FIELD_H__
#define __FDAPDE_SCALAR_FIELD_H__

#include "header_check.h"

namespace fdapde {

// forward decl
template <int StaticInputSize, typename Derived> struct ScalarFieldBase;
template <typename Derived> class Gradient;
template <typename Derived> class Hessian;
template <typename Derived> class Laplacian;

template <typename Derived_, typename UnaryFunctor>
struct ScalarFieldUnaryOp :
    public ScalarFieldBase<Derived_::StaticInputSize, ScalarFieldUnaryOp<Derived_, UnaryFunctor>> {
    using Derived = Derived_;
    template <typename T> using Meta = ScalarFieldUnaryOp<T, UnaryFunctor>;
    using Base = ScalarFieldBase<Derived::StaticInputSize, ScalarFieldUnaryOp<Derived, UnaryFunctor>>;
    using InputType = typename Derived::InputType;
    using Scalar = decltype(std::declval<UnaryFunctor>().operator()(std::declval<typename Derived::Scalar>()));
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;

    constexpr ScalarFieldUnaryOp(const Derived_& derived, const UnaryFunctor& op) :
        Base(), derived_(derived), op_(op) { }
    constexpr ScalarFieldUnaryOp(const Derived_& derived) : ScalarFieldUnaryOp(derived, UnaryFunctor {}) { }
    constexpr Scalar operator()(const InputType& p) const {
        if constexpr (StaticInputSize == Dynamic) { fdapde_assert(p.rows() == Base::input_size()); }
        return op_(derived_(p));
    }
    constexpr int input_size() const { return derived_.input_size(); }
    constexpr const Derived& derived() const { return derived_; }
   private:
    internals::ref_select_t<const Derived> derived_;
    UnaryFunctor op_;
};
template <int Size, typename Derived> constexpr auto sin(const ScalarFieldBase<Size, Derived>& f) {
    return ScalarFieldUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::sin(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto cos(const ScalarFieldBase<Size, Derived>& f) {
    return ScalarFieldUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::cos(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto logn(const ScalarFieldBase<Size, Derived>& f) {
    return ScalarFieldUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::log(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto exp(const ScalarFieldBase<Size, Derived>& f) {
    return ScalarFieldUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::exp(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto abs(const ScalarFieldBase<Size, Derived>& f) {
    return ScalarFieldUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::abs(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto sqrt(const ScalarFieldBase<Size, Derived>& f) {
    return ScalarFieldUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::sqrt(x); })>(f.derived());
}

namespace internals {
class pow_t {
    int i_ = 0;
   public:
    constexpr explicit pow_t(int i) : i_(i) { }
    template <typename Scalar> constexpr Scalar operator()(Scalar&& t) const { return fdapde::pow(t, i_); }
};
}   // namespace internals

template <int Size, typename Derived> constexpr auto pow(const ScalarFieldBase<Size, Derived>& f, int i) {
    return ScalarFieldUnaryOp<Derived, internals::pow_t>(f.derived(), internals::pow_t(i));
}

template <typename Lhs, typename Rhs, typename BinaryOperation>
class ScalarFieldBinOp : public ScalarFieldBase<Lhs::StaticInputSize, ScalarFieldBinOp<Lhs, Rhs, BinaryOperation>> {
    fdapde_static_assert(
      Lhs::StaticInputSize == Rhs::StaticInputSize, YOU_MIXED_SCALAR_FUNCTIONS_WITH_DIFFERENT_STATIC_INNER_SIZES);
    fdapde_static_assert(
      std::is_convertible_v<typename Lhs::Scalar FDAPDE_COMMA typename Rhs::Scalar>,
      YOU_MIXED_SCALAR_FIELDS_WITH_NON_CONVERTIBLE_SCALAR_OUTPUT_TYPES);
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = ScalarFieldBinOp<T1, T2, BinaryOperation>;
    using Base =
      ScalarFieldBase<LhsDerived::StaticInputSize, ScalarFieldBinOp<LhsDerived, RhsDerived, BinaryOperation>>;
    using LhsInputType = typename LhsDerived::InputType;
    using RhsInputType = typename RhsDerived::InputType;
    using InputType = internals::prefer_most_derived_t<LhsInputType, RhsInputType>;
    using Scalar = typename LhsDerived::Scalar;
    static constexpr int StaticInputSize = LhsDerived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = LhsDerived::XprBits | RhsDerived::XprBits;

    ScalarFieldBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) requires(StaticInputSize == Dynamic) :
        Base(), lhs_(lhs), rhs_(rhs), op_(op) {
        fdapde_assert(lhs.input_size() == rhs.input_size());
    }
    constexpr ScalarFieldBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op)
        requires(StaticInputSize != Dynamic)
        : Base(), lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr ScalarFieldBinOp(const Lhs& lhs, const Rhs& rhs) : ScalarFieldBinOp(lhs, rhs, BinaryOperation {}) { }
    constexpr Scalar operator()(const InputType& p) const {
        fdapde_static_assert(
          std::is_same_v<LhsInputType FDAPDE_COMMA RhsInputType> ||
            internals::are_related_by_inheritance_v<LhsInputType FDAPDE_COMMA RhsInputType>,
          YOU_MIXED_SCALAR_FIELDS_WITH_INCOMPATIBLE_INPUT_TYPES);
        if constexpr (StaticInputSize == Dynamic) { fdapde_assert(p.rows() == Base::input_size()); }
        return op_(lhs_(p), rhs_(p));
    }
    constexpr int input_size() const { return lhs_.input_size(); }
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
   private:
    internals::ref_select_t<const LhsDerived> lhs_;
    internals::ref_select_t<const RhsDerived> rhs_;
    BinaryOperation op_;
};

template <int Size, typename Lhs, typename Rhs>
constexpr ScalarFieldBinOp<Lhs, Rhs, std::plus<>>
operator+(const ScalarFieldBase<Size, Lhs>& lhs, const ScalarFieldBase<Size, Rhs>& rhs) {
    return ScalarFieldBinOp<Lhs, Rhs, std::plus<>> {lhs.derived(), rhs.derived(), std::plus<>()};
}
template <int Size, typename Lhs, typename Rhs>
constexpr ScalarFieldBinOp<Lhs, Rhs, std::minus<>>
operator-(const ScalarFieldBase<Size, Lhs>& lhs, const ScalarFieldBase<Size, Rhs>& rhs) {
    return ScalarFieldBinOp<Lhs, Rhs, std::minus<>> {lhs.derived(), rhs.derived(), std::minus<>()};
}
template <int Size, typename Lhs, typename Rhs>
constexpr ScalarFieldBinOp<Lhs, Rhs, std::multiplies<>>
operator*(const ScalarFieldBase<Size, Lhs>& lhs, const ScalarFieldBase<Size, Rhs>& rhs) {
    return ScalarFieldBinOp<Lhs, Rhs, std::multiplies<>> {lhs.derived(), rhs.derived(), std::multiplies<>()};
}
template <int Size, typename Lhs, typename Rhs>
constexpr ScalarFieldBinOp<Lhs, Rhs, std::divides<>>
operator/(const ScalarFieldBase<Size, Lhs>& lhs, const ScalarFieldBase<Size, Rhs>& rhs) {
    return ScalarFieldBinOp<Lhs, Rhs, std::divides<>> {lhs.derived(), rhs.derived(), std::divides<>()};
}

template <typename Lhs_, typename Rhs_, typename BinaryOperation>
struct ScalarFieldCoeffOp :
    public ScalarFieldBase<
      std::conditional_t<std::is_arithmetic_v<Lhs_>, Rhs_, Lhs_>::StaticInputSize,
      ScalarFieldCoeffOp<Lhs_, Rhs_, BinaryOperation>> {
   private:
    // keep this private to avoid to consider ScalarFieldCoeffOp as a unary node
    using Derived = std::conditional_t<std::is_arithmetic_v<Lhs_>, Rhs_, Lhs_>;
   public:
    using CoeffType = std::conditional_t<std::is_arithmetic_v<Lhs_>, Lhs_, Rhs_>;
    fdapde_static_assert(
      std::is_convertible_v<CoeffType FDAPDE_COMMA typename Derived::Scalar> && std::is_arithmetic_v<CoeffType>,
      COEFFICIENT_IN_BINARY_OPERATION_NOT_CONVERTIBLE_TO_SCALAR_TYPE);
    using LhsDerived = Lhs_;
    using RhsDerived = Rhs_;
    template <typename T1, typename T2> using Meta = ScalarFieldCoeffOp<T1, T2, BinaryOperation>;
    static constexpr bool is_coeff_lhs =
      std::is_arithmetic_v<Lhs_>;   // whether to perform op_(xpr_(p), coeff) or op_(coeff, xpr_(p))
    using Base = ScalarFieldBase<Derived::StaticInputSize, ScalarFieldCoeffOp<LhsDerived, RhsDerived, BinaryOperation>>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;

    constexpr ScalarFieldCoeffOp(const Lhs_& lhs, const Rhs_& rhs, BinaryOperation op) :
        Base(), lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr ScalarFieldCoeffOp(const Lhs_& lhs, const Rhs_& rhs) :
        ScalarFieldCoeffOp(lhs, rhs, BinaryOperation {}) { }
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
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
   private:
    internals::ref_select_t<const LhsDerived> lhs_;
    internals::ref_select_t<const RhsDerived> rhs_;
    BinaryOperation op_;
};

#define FDAPDE_DEFINE_SCALAR_COEFF_OP(OPERATOR, FUNCTOR)                                                               \
    template <int Size, typename Derived, typename Coeff>                                                              \
    constexpr ScalarFieldCoeffOp<Derived, Coeff, FUNCTOR> OPERATOR(                                                    \
      const ScalarFieldBase<Size, Derived>& lhs, Coeff rhs)                                                            \
        requires(std::is_arithmetic_v<Coeff>)                                                                          \
    {                                                                                                                  \
        return ScalarFieldCoeffOp<Derived, Coeff, FUNCTOR> {lhs.derived(), rhs, FUNCTOR()};                            \
    }                                                                                                                  \
    template <int Size, typename Derived, typename Coeff>                                                              \
    constexpr ScalarFieldCoeffOp<Coeff, Derived, FUNCTOR> OPERATOR(                                                    \
      Coeff lhs, const ScalarFieldBase<Size, Derived>& rhs)                                                            \
        requires(std::is_arithmetic_v<Coeff>)                                                                          \
    {                                                                                                                  \
        return ScalarFieldCoeffOp<Coeff, Derived, FUNCTOR> {lhs, rhs.derived(), FUNCTOR()};                            \
    }

FDAPDE_DEFINE_SCALAR_COEFF_OP(operator+, std::plus<>      )
FDAPDE_DEFINE_SCALAR_COEFF_OP(operator-, std::minus<>     )
FDAPDE_DEFINE_SCALAR_COEFF_OP(operator*, std::multiplies<>)
FDAPDE_DEFINE_SCALAR_COEFF_OP(operator/, std::divides<>   )

template <
  int Size,   // input space dimension (Dynamic accepted)
  typename FunctorType_ = std::function<double(internals::static_dynamic_eigen_vector_selector_t<Size>)>>
class ScalarField : public ScalarFieldBase<Size, ScalarField<Size, FunctorType_>> {
    using FunctorType = std::decay_t<FunctorType_>;   // type of wrapped functor
    using traits = internals::fn_ptr_traits<&FunctorType::operator()>;
    fdapde_static_assert(traits::n_args == 1, PROVIDED_FUNCTOR_MUST_ACCEPT_ONLY_ONE_ARGUMENT);
   public:
    using Base = ScalarFieldBase<Size, ScalarField<Size, FunctorType>>;
    using InputType = std::tuple_element_t<0, typename traits::ArgsType>;
    using Scalar = typename std::invoke_result<FunctorType, InputType>::type;
    static constexpr int StaticInputSize = Size;      // dimensionality of base space (can be Dynamic)
    static constexpr int NestAsRef = 0;               // whether to store the node by reference of by copy
    static constexpr int XprBits = 0;                 // bits which carries implementation specific informations

    constexpr ScalarField() : f_() { }
    constexpr explicit ScalarField(int n)
        requires(StaticInputSize == Dynamic)
        : Base(), f_(), dynamic_input_size_(n) { }
    constexpr explicit ScalarField(const FunctorType& f) : f_(f) {};
    template <typename Expr> ScalarField(const ScalarFieldBase<Size, Expr>& f) {
        fdapde_static_assert(
          std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(InputType)>> &&
            std::is_convertible_v<typename Expr::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_FUNCTION_ASSIGNMENT);
        Expr expr = f.get();
        f_ = [expr](const InputType& x) { return expr(x); };
    }
    template <typename Expr> ScalarField& operator=(const ScalarFieldBase<Size, Expr>& f) {
        fdapde_static_assert(
          std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(InputType)>> &&
            std::is_convertible_v<typename Expr::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_FUNCTION_ASSIGNMENT);
        Expr expr = f.get();
        f_ = [expr](const InputType& x) { return expr(x); };
        return *this;
    }
    // assignment from lambda expression
    template <typename LamdaType> ScalarField& operator=(const LamdaType& lambda) {
        fdapde_static_assert(
          std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(InputType)>> &&
            std::is_convertible_v<
              typename std::invoke_result<LamdaType FDAPDE_COMMA InputType>::type FDAPDE_COMMA Scalar>,
          INVALID_LAMBDA_EXPRESSION_ASSIGNMENT);
        f_ = lambda;
        return *this;
    }
    // static initializers
    struct ConstantField : public ScalarFieldBase<Size, ConstantField> {
        Scalar c_ = 0;
        ConstantField(Scalar c) : c_(c) { }
        constexpr Scalar operator()([[maybe_unused]] const InputType& x) const { return c_; }
    };
    static constexpr ConstantField Constant(Scalar c) { return ConstantField(c); }
    static constexpr ConstantField Zero() { return ConstantField(0.0); }
    constexpr int input_size() const { return StaticInputSize == Dynamic ? dynamic_input_size_ : StaticInputSize; }
    // evaluation at point
    constexpr Scalar operator()(const InputType& x) const { return f_(x); }
    constexpr Scalar operator()(const InputType& x) { return f_(x); }
    // evaluation at matrix of points
    Eigen::Matrix<Scalar, Dynamic, 1> eval_at(const Eigen::Matrix<double, Dynamic, Dynamic>& points) const {
        fdapde_static_assert(
          std::is_invocable_v<FunctorType FDAPDE_COMMA decltype(points.row(std::declval<int>()))>,
          INVALID_SCALAR_FIELD_INVOCATION);
        fdapde_assert(points.rows() > 0 && points.cols() == input_size());
	Eigen::Matrix<Scalar, Dynamic, 1> evals(points.rows());
        for (int i = 0; i < points.rows(); ++i) { evals[i] = f_(points.row(i)); }
        return evals;
    }
    void resize(int dynamic_input_size) {
        fdapde_static_assert(StaticInputSize == Dynamic, YOU_CALLED_A_DYNAMIC_METHOD_ON_A_STATIC_SIZED_FIELD);
        dynamic_input_size_ = dynamic_input_size;
    }
   protected:
    FunctorType f_ {};
    int dynamic_input_size_ = 0;   // run-time base space dimension
};

template <typename Derived, int Order> struct PartialDerivative;

template <typename Derived_>
struct PartialDerivative<Derived_, 1> :
    public ScalarFieldBase<Derived_::StaticInputSize, PartialDerivative<Derived_, 1>> {
    using Derived = Derived_;
    template <typename T> using Meta = PartialDerivative<T, 1>;
    using Base = ScalarFieldBase<Derived::StaticInputSize, PartialDerivative<Derived, 1>>;
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
    internals::ref_select_t<Derived> f_;
    int i_;
    double h_ = 1e-3;
};

template <typename Derived_>
struct PartialDerivative<Derived_, 2> :
    public ScalarFieldBase<Derived_::StaticInputSize, PartialDerivative<Derived_, 2>> {
    using Derived = Derived_;
    template <typename T> using Meta = PartialDerivative<T, 2>;
    using Base = ScalarFieldBase<Derived::StaticInputSize, PartialDerivative<Derived, 2>>;
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
    internals::ref_select_t<Derived> f_;
    int i_, j_;
    double h_ = 1e-3;
};

template <int Size, typename Derived> struct ScalarFieldBase {
    constexpr ScalarFieldBase() = default;
  
    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    constexpr auto operator-() const {   // unary minus
        return ScalarFieldUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return -x; })>(derived());
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
    // compatibility with matrix expressions (these are here just to let write code which handles both the scalar and
    // vector case transparently)
    static constexpr int Rows = 1;
    static constexpr int Cols = 1;
    template <typename Rhs> constexpr auto dot(const ScalarFieldBase<Size, Rhs>& rhs) const;
   protected:
    double step_ = 1e-3;   // step size used in derivative approximation
};

#ifdef __FDAPDE_HAS_EIGEN__
// special fields
template <int StaticInputSize>
struct ZeroField : public ScalarField<StaticInputSize, decltype([](const Eigen::Matrix<double, StaticInputSize, 1>&) {
                                          return 0.0;
                                      })> { };
#endif
  
namespace internals {

// for type Functor_ having just a call operator, xpr_wrap<Xpr_, Functor_> makes Functor_ a valid expression type to be
// stored inside Xpr_
template <int StaticInputSize_, typename Functor_, int Bits_ = 0>
class xpr_scalar_wrap : ScalarFieldBase<StaticInputSize_, xpr_scalar_wrap<StaticInputSize_, Functor_, Bits_>> {
    using FunctorType = std::decay_t<Functor_>;
    using fn_traits = internals::fn_ptr_traits<&FunctorType::operator()>;
    fdapde_static_assert(std::tuple_size_v<typename fn_traits::ArgsType> == 1, FUNCTOR_MUST_ACCEPT_A_SINGLE_ARGUMENT);
    FunctorType f_;
   public:
    using InputType = std::tuple_element_t<0, typename fn_traits::ArgsType>;
    using Scalar = typename fn_traits::RetType;
    static constexpr int StaticInputSize = StaticInputSize_;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | int(Bits_);
    static constexpr int Rows = 1;
    static constexpr int Cols = 1;

    constexpr xpr_scalar_wrap() noexcept : f_() { }
    constexpr xpr_scalar_wrap(const Functor_& f) noexcept : f_(f) { }
    // accessors
    constexpr Scalar operator()(const InputType& p) const { return f_(p); }
    constexpr int input_size() const { return StaticInputSize; }
};
template <typename Xpr_, int StaticInputSize_, typename Functor_, int Bits_ = 0>
struct xpr_or_scalar_wrap :
    std::type_identity<std::conditional_t<
      internals::is_scalar_field_v<Xpr_>, Xpr_, xpr_scalar_wrap<StaticInputSize_, Functor_, Bits_>>> { };
template <typename Xpr_, int StaticInputSize_, typename Functor_, int Bits_ = 0>
using xpr_or_scalar_wrap_t = typename xpr_or_scalar_wrap<Xpr_, StaticInputSize_, Functor_, Bits_>::type;

}   // namespace internals

}   // namespace fdapde

#endif   // __FDAPDE_SCALAR_FIELD_H__
