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

#ifndef __FE_FUNCTION_H__
#define __FE_FUNCTION_H__

#include "../fields/scalar_field.h"
#include "fe_assembler.h"

namespace fdapde {

  // still missing evaluation of partial derivatives
  
template <int StaticInputSize, typename Derived> struct FeFunctionBase;
  
// representation of u(x) = \sum_{i=1}^{n_dofs} u_i \psi_i(x) with \{ \psi_i \}_i a finite element basis
template <typename FeSpace_> class FeFunction : public FeFunctionBase<FeSpace_::local_dim, FeFunction<FeSpace_>> {
    using Triangulation = typename std::decay_t<FeSpace_>::Triangulation;
   public:
    using FeSpace = std::decay_t<FeSpace_>;
    using Base = ScalarBase<FeSpace::local_dim, FeFunction<FeSpace>>;
    using InputType = SVector<FeSpace::local_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = FeSpace::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int XprBits = 0;

    FeFunction() = default;
    FeFunction(FeSpace_& fe_space) : fe_space_(&fe_space), triangulation_(&fe_space.triangulation()) {
        coeff_ = DVector<double>::Zero(fe_space_->n_dofs());
    }
    FeFunction(FeSpace_& fe_space, const DVector<double>& coeff) :
        fe_space_(&fe_space), triangulation_(&fe_space.triangulation()), coeff_(coeff) {
        fdapde_assert(coeff.size() > 0 && coeff.size() == fe_space_->n_dofs());
    }
    // point p already mapped on reference cell
    constexpr Scalar eval(const InputType& p, const DVector<int>& active_dofs) const {
        Scalar value = 0;
        for (int i = 0; i < fe_space().n_basis(); ++i) { value += coeff_[active_dofs[i]] * fe_space().eval(i, p); }
        return value;
    }
    // norms of fe functions
    double l2_squared_norm() {
        internals::fe_assembler_mass_loop<DofHandler<local_dim, embed_dim>, typename FeSpace::FeType> assembler(
          fe_space_->dof_handler());
        return coeff_.dot(assembler.run() * coeff_);
    }
    double l2_norm() { return std::sqrt(l2_squared_norm()); }
    double h1_squared_norm() const;   // implementation below
    double h1_norm() const { return std::sqrt(h1_squared_norm()); }

    // getters
    const DVector<double>& coeff() const { return coeff_; }
    constexpr FeSpace& fe_space() { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }
    DVector<double>& coeff() { return coeff_; }   // non-const access to expansion coefficient vector
   private:
    DVector<double> coeff_;
    FeSpace* fe_space_;
    const Triangulation* triangulation_;
};

template <typename Lhs_, typename Rhs_, typename BinaryOperation>
struct FeFunctionBinOp : public FeFunctionBase<Lhs_::StaticInputSize, FeFunctionBinOp<Lhs_, Rhs_, BinaryOperation>> {
    fdapde_static_assert(
        std::is_same_v<typename Lhs_::FeSpace FDAPDE_COMMA typename Rhs_::FeSpace>,
      EITHER_LHS_OR_RHS_IS_A_TRIAL_FUNCTION_OR_THEY_MUST_BE_DEFINED_ON_THE_SAME_FINITE_ELEMENT_SPACE);
    using LhsDerived = Lhs_;
    using RhsDerived = Rhs_;
    using Base = FeFunctionBase<LhsDerived::StaticInputSize, FeFunctionBinOp<LhsDerived, RhsDerived, BinaryOperation>>;
    using Scalar = typename LhsDerived::Scalar;
    using InputType = typename LhsDerived::InputType;
    using FeSpace = typename LhsDerived::FeSpace;
    static constexpr int StaticInputSize = LhsDerived::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int local_dim = LhsDerived::local_dim;
    static constexpr int embed_dim = LhsDerived::embed_dim;
    static constexpr int XprBits = LhsDerived::XprBits | RhsDerived::XprBits;

    constexpr FeFunctionBinOp(const Lhs_& lhs, const RhsDerived& rhs, const BinaryOperation& op) :
        Base(), lhs_(lhs), rhs_(rhs), op_(op) { }
    // point p already mapped on reference cell
    constexpr Scalar eval(const InputType& p, const DVector<int>& active_dofs) const {
        return op_(lhs_.eval(p, active_dofs), rhs_.eval(p, active_dofs));
    }
    typename LhsDerived::FeSpace& fe_space() { return lhs_.fe_space(); }
    constexpr int input_size() const { return lhs_.input_size(); }
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
   private:
    typename internals::ref_select<const LhsDerived>::type lhs_;
    typename internals::ref_select<const RhsDerived>::type rhs_;
    BinaryOperation op_;
};

template <int Size, typename Lhs, typename Rhs>
constexpr FeFunctionBinOp<Lhs, Rhs, std::plus<>>
operator+(const FeFunctionBase<Size, Lhs>& lhs, const FeFunctionBase<Size, Rhs>& rhs) {
    return FeFunctionBinOp<Lhs, Rhs, std::plus<>> {lhs.derived(), rhs.derived(), std::plus<>()};
}
template <int Size, typename Lhs, typename Rhs>
constexpr FeFunctionBinOp<Lhs, Rhs, std::minus<>>
operator-(const FeFunctionBase<Size, Lhs>& lhs, const FeFunctionBase<Size, Rhs>& rhs) {
    return FeFunctionBinOp<Lhs, Rhs, std::minus<>> {lhs.derived(), rhs.derived(), std::minus<>()};
}
template <int Size, typename Lhs, typename Rhs>
constexpr FeFunctionBinOp<Lhs, Rhs, std::multiplies<>>
operator*(const FeFunctionBase<Size, Lhs>& lhs, const FeFunctionBase<Size, Rhs>& rhs) {
    return FeFunctionBinOp<Lhs, Rhs, std::multiplies<>> {lhs.derived(), rhs.derived(), std::multiplies<>()};
}
template <int Size, typename Lhs, typename Rhs>
constexpr FeFunctionBinOp<Lhs, Rhs, std::divides<>>
operator/(const FeFunctionBase<Size, Lhs>& lhs, const FeFunctionBase<Size, Rhs>& rhs) {
    return FeFunctionBinOp<Lhs, Rhs, std::divides<>> {lhs.derived(), rhs.derived(), std::divides<>()};
}

template <typename Derived_, typename UnaryFunctor>
class FeFunctionUnaryOp : public FeFunctionBase<Derived_::StaticInputSize, FeFunctionUnaryOp<Derived_, UnaryFunctor>> {
   public:
    using Derived = Derived_;
    using Base = FeFunctionBase<Derived::StaticInputSize, FeFunctionUnaryOp<Derived, UnaryFunctor>>;
    using InputType = typename Derived::InputType;
    using Scalar = decltype(std::declval<UnaryFunctor>().operator()(std::declval<typename Derived::Scalar>()));
    using FeSpace = typename Derived::FeSpace;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int local_dim = Derived::local_dim;
    static constexpr int embed_dim = Derived::embed_dim;
    static constexpr int XprBits = Derived::XprBits;
  
    constexpr FeFunctionUnaryOp(const Derived& derived, const UnaryFunctor& op) : Base(), derived_(derived), op_(op) { }
    constexpr FeFunctionUnaryOp(const Derived& derived) : FeFunctionUnaryOp(derived, UnaryFunctor()) { }
    // point p already mapped on reference cell  
    constexpr Scalar eval(const InputType& p, const DVector<int>& active_dofs) const {
        return op_(derived_.eval(p, active_dofs));
    }
    typename Derived::FeSpace& fe_space() { return derived_.fe_space(); }
    constexpr int input_size() const { return derived_.input_size(); }
    constexpr const Derived& derived() const { return derived_; }
   private:
    typename internals::ref_select<const Derived>::type derived_;
    UnaryFunctor op_;
};
template <int Size, typename Derived> constexpr auto sin(const FeFunctionBase<Size, Derived>& f) {
    return FeFunctionUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::sin(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto cos(const FeFunctionBase<Size, Derived>& f) {
    return FeFunctionUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::cos(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto logn(const FeFunctionBase<Size, Derived>& f) {
    return FeFunctionUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::log(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto exp(const FeFunctionBase<Size, Derived>& f) {
    return FeFunctionUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::exp(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto abs(const FeFunctionBase<Size, Derived>& f) {
    return FeFunctionUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::abs(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto sqrt(const FeFunctionBase<Size, Derived>& f) {
    return FeFunctionUnaryOp<Derived, decltype([](typename Derived::Scalar x) { return std::sqrt(x); })>(f.derived());
}
template <int Size, typename Derived> constexpr auto pow(const FeFunctionBase<Size, Derived>& f, int i) {
    return FeFunctionUnaryOp<Derived, internals::pow_t>(f.derived(), internals::pow_t(i));
}
  
template <typename Lhs, typename Rhs, typename BinaryOperation>
struct FeFunctionCoeffOp :
    public FeFunctionBase<
      std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>::StaticInputSize,
      FeFunctionCoeffOp<Lhs, Rhs, BinaryOperation>> {
    using CoeffType = std::conditional_t<std::is_arithmetic_v<Lhs>, Lhs, Rhs>;
    using Derived = std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>;
    fdapde_static_assert(
      std::is_convertible_v<CoeffType FDAPDE_COMMA typename Derived::Scalar> && std::is_arithmetic_v<CoeffType>,
      COEFFICIENT_IN_BINARY_OPERATION_NOT_CONVERTIBLE_TO_SCALAR_TYPE);
    using Base = FeFunctionBase<Derived::StaticInputSize, FeFunctionCoeffOp<Lhs, Rhs, BinaryOperation>>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    using FeSpace = typename Derived::FeSpace;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;
    static constexpr int local_dim = Derived::local_dim;
    static constexpr int embed_dim = Derived::embed_dim;

    constexpr FeFunctionCoeffOp(const Derived& derived, CoeffType coeff, BinaryOperation op) :
        Base(), derived_(derived), coeff_(coeff), op_(op) { }
    // point p already mapped on reference cell
    constexpr Scalar eval(const InputType& p, const DVector<int>& active_dofs) const {
        if constexpr (is_coeff_lhs) {
            return op_(coeff_, derived_.eval(p, active_dofs));
        } else {
            return op_(derived_.eval(p, active_dofs), coeff_);
        }
    }
    FeSpace& fe_space() { return derived_.fe_space(); }
    constexpr int input_size() const { return derived_.input_size(); }
    constexpr const Derived& derived() const { return derived_; }
   private:
    typename internals::ref_select<const Derived>::type derived_;
    BinaryOperation op_;
    CoeffType coeff_;
    static constexpr bool is_coeff_lhs =
      std::is_arithmetic_v<Lhs>;   // whether to perform op_(xpr_(p), coeff) or op_(coeff, xpr_(p))
};

#define FDAPDE_DEFINE_FE_FUNCTION_COEFF_OP(OPERATOR, FUNCTOR)                                                          \
    template <int Size, typename Derived, typename Coeff>                                                              \
    constexpr FeFunctionCoeffOp<Derived, Coeff, FUNCTOR> OPERATOR(const FeFunctionBase<Size, Derived>& lhs, Coeff rhs) \
      requires(std::is_arithmetic_v<Coeff>) {                                                                          \
        return FeFunctionCoeffOp<Derived, Coeff, FUNCTOR> {lhs.derived(), rhs, FUNCTOR()};                             \
    }                                                                                                                  \
    template <int Size, typename Derived, typename Coeff>                                                              \
    constexpr FeFunctionCoeffOp<Coeff, Derived, FUNCTOR> OPERATOR(Coeff lhs, const FeFunctionBase<Size, Derived>& rhs) \
        requires(std::is_arithmetic_v<Coeff>) {                                                                        \
        return FeFunctionCoeffOp<Coeff, Derived, FUNCTOR> {rhs.derived(), lhs, FUNCTOR()};                             \
    }

FDAPDE_DEFINE_FE_FUNCTION_COEFF_OP(operator+, std::plus<>      )
FDAPDE_DEFINE_FE_FUNCTION_COEFF_OP(operator-, std::minus<>     )
FDAPDE_DEFINE_FE_FUNCTION_COEFF_OP(operator*, std::multiplies<>)
FDAPDE_DEFINE_FE_FUNCTION_COEFF_OP(operator/, std::divides<>   )

template <int Size, typename Derived> struct FeFunctionBase : public fdapde::ScalarBase<Size, Derived> {
    constexpr FeFunctionBase() = default;

    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }

    template <typename InputType> constexpr auto operator()(const InputType& p) const {
        fdapde_static_assert(
          std::is_same_v<typename Derived::InputType FDAPDE_COMMA InputType>,
          CANNOT_EVALUATE_FINITE_ELEMENT_EXPRESSION_WITH_THIS_INPUT_TYPE);
        int e_id = derived().fe_space().triangulation().locate(p);
	// return NaN if point lies outside domain
        if (e_id == -1) return std::numeric_limits<typename Derived::Scalar>::quiet_NaN();
	// map p to reference cell and evaluate
        typename DofHandler<Derived::local_dim, Derived::embed_dim>::CellType cell =
          derived().fe_space().dof_handler().cell(e_id);
        InputType ref_p = cell.invJ() * (p - cell.node(0));
        return derived().eval(ref_p, cell.dofs());
    }
};

// implementation of trial and test functions
  
template <typename FeSpace_>
struct TestFunction : public fdapde::ScalarBase<FeSpace_::local_dim, TestFunction<FeSpace_>> {
    using TestSpace = std::decay_t<FeSpace_>;
    using InputType = internals::fe_assembler_packet<TestSpace::local_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = TestSpace::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | bilinear_bits::compute_shape_values;
    static constexpr int local_dim = TestSpace::local_dim;
    static constexpr int embed_dim = TestSpace::embed_dim;
  
    constexpr TestFunction() = default;
    constexpr TestFunction(FeSpace_& fe_space) : fe_space_(&fe_space) { }
    constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_value; }
    constexpr TestSpace& fe_space() { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }
   private:
    TestSpace* fe_space_;
};

template <typename FeSpace_>
struct PartialDerivative<TestFunction<FeSpace_>, 1> :
    public fdapde::ScalarBase<FeSpace_::local_dim, PartialDerivative<TestFunction<FeSpace_>, 1>> {
    using Derived = TestFunction<FeSpace_>;
    using InputType = internals::fe_assembler_packet<FeSpace_::local_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = FeSpace_::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | bilinear_bits::compute_shape_grad;

    PartialDerivative() = default;
    PartialDerivative(const Derived& f, int i) : f_(f), i_(i) { }
    // fe assembly evaluation
    constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_grad[i_]; }
    constexpr int input_size() const { return f_.input_size(); }
   private:
    typename internals::ref_select<Derived>::type f_;
    int i_;
};
  
template <typename FeSpace_>
struct TrialFunction : public fdapde::ScalarBase<FeSpace_::local_dim, TrialFunction<FeSpace_>> {
    using TrialSpace = std::decay_t<FeSpace_>;
    using InputType = internals::fe_assembler_packet<TrialSpace::local_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = TrialSpace::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | bilinear_bits::compute_shape_values;
    static constexpr int local_dim = TrialSpace::local_dim;
    static constexpr int embed_dim = TrialSpace::embed_dim;

    constexpr TrialFunction() = default;
    constexpr TrialFunction(FeSpace_& fe_space) : fe_space_(&fe_space) { }

    double l2_squared_norm() {
        internals::fe_assembler_mass_loop<DofHandler<local_dim, embed_dim>, typename TrialSpace::FeType> assembler(
          fe_space_->dof_handler());
        return coeff_.dot(assembler.run() * coeff_);
    }
    double l2_norm() { return std::sqrt(l2_squared_norm()); }
    double h1_squared_norm() const {
        TrialFunction u(*fe_space_);
        TestFunction  v(*fe_space_);
        auto assembler = integrate(*fe_space_->triangulation())(u * v + inner(grad(u), grad(v)));
        return coeff_.dot(assembler.run() * coeff_);
    }
    double h1_norm() const { return std::sqrt(h1_squared_norm()); }

    // fe assembler evaluation
    constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.trial_value; }
    // getters
    constexpr TrialSpace& fe_space() { return *fe_space_; }
    const DVector<double>& coeff() const { return coeff_; }
    constexpr int input_size() const { return StaticInputSize; }
    DVector<double>& coeff() { return coeff_; }   // non-const access to expansion coefficient vector
   private:
    DVector<double> coeff_;
    TrialSpace* fe_space_;
};

template <typename FeSpace_>
struct PartialDerivative<TrialFunction<FeSpace_>, 1> :
    public fdapde::ScalarBase<FeSpace_::local_dim, PartialDerivative<TrialFunction<FeSpace_>, 1>> {
    using Derived = TrialFunction<FeSpace_>;
    using Base = ScalarBase<FeSpace_::local_dim, PartialDerivative<Derived, 1>>;
    using InputType = internals::fe_assembler_packet<FeSpace_::local_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = FeSpace_::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | bilinear_bits::compute_shape_grad;

    PartialDerivative() = default;
    PartialDerivative(const Derived& f, int i) : Base(), f_(f), i_(i) { }
    // fe assembly evaluation
    constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.trial_grad[i_]; }
    constexpr int input_size() const { return f_.input_size(); }
   private:
    typename internals::ref_select<Derived>::type f_;
    int i_;
};

// Sobolev H^1 norm of finite element function
template <typename FeSpace_> double FeFunction<FeSpace_>::h1_squared_norm() const {
    TrialFunction u(*fe_space_);
    TestFunction  v(*fe_space_);
    auto assembler = integrate(*triangulation_)(u * v + inner(grad(u), grad(v)));
    return coeff_.dot(assembler.run() * coeff_);
}
  
}   // namespace fdapde

#endif   // __FE_FUNCTION_H__
