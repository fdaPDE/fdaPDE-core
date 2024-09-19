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

template <typename FeSpace_> struct TestFunction;
template <typename FeSpace_> struct TrialFunction;

namespace internals {

template <typename FeSpace_>
struct scalar_test_function_impl : public fdapde::ScalarBase<FeSpace_::local_dim, TestFunction<FeSpace_>> {
    using TestSpace = std::decay_t<FeSpace_>;
    using Base = fdapde::ScalarBase<FeSpace_::local_dim, TestFunction<FeSpace_>>;
    using InputType = internals::fe_assembler_packet<TestSpace::local_dim, TestSpace::n_components>;
    using Scalar = double;
    static constexpr int StaticInputSize = TestSpace::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | bilinear_bits::compute_shape_values;

    constexpr scalar_test_function_impl() = default;
    constexpr scalar_test_function_impl(FeSpace_& fe_space) : fe_space_(&fe_space) { }

    struct FirstPartialDerivative : fdapde::ScalarBase<TestSpace::local_dim, FirstPartialDerivative> {
        using Derived = TestFunction<FeSpace_>;
        using InputType = internals::fe_assembler_packet<TestSpace::local_dim, TestSpace::n_components>;
        using Scalar = double;
        static constexpr int StaticInputSize = TestSpace::local_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | bilinear_bits::compute_shape_grad;

        FirstPartialDerivative() = default;
        FirstPartialDerivative([[maybe_unused]] const Derived& f, int i) : i_(i) { }
        // fe assembly evaluation
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_grad[i_]; }
        constexpr int input_size() const { return StaticInputSize; }
       private:
        int i_;
    };
    // fe assembly evaluation
    constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_value[0]; }
    constexpr TestSpace& fe_space() { return *fe_space_; }
    constexpr const TestSpace& fe_space() const { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }
   private:
    TestSpace* fe_space_;
};

template <typename FeSpace_>
struct vector_test_function_impl : public fdapde::MatrixBase<FeSpace_::local_dim, TestFunction<FeSpace_>> {
    using TestSpace = std::decay_t<FeSpace_>;
    using Base = fdapde::MatrixBase<FeSpace_::local_dim, TestFunction<FeSpace_>>;
    using InputType = internals::fe_assembler_packet<TestSpace::local_dim, TestSpace::n_components>;
    using Scalar = double;
    static constexpr int StaticInputSize = TestSpace::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | bilinear_bits::compute_shape_values;
    static constexpr int ReadOnly = 1;
    static constexpr int Rows = FeSpace_::n_components;
    static constexpr int Cols = 1;

    constexpr vector_test_function_impl() = default;
    constexpr vector_test_function_impl(FeSpace_& fe_space) : fe_space_(&fe_space) { }

    struct Jacobian : fdapde::MatrixBase<TestSpace::local_dim, Jacobian> {
        using Derived = TestFunction<FeSpace_>;
        using InputType = internals::fe_assembler_packet<TestSpace::local_dim, TestSpace::n_components>;
        using Scalar = double;
        static constexpr int StaticInputSize = TestSpace::local_dim;
        static constexpr int Rows = TestSpace::local_dim;
        static constexpr int Cols = TestSpace::n_components;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | bilinear_bits::compute_shape_grad;

        Jacobian() = default;
        // fe assembly evaluation
        constexpr Scalar eval(int i, int j, const InputType& fe_packet) const { return fe_packet.test_grad(i, j); }
        constexpr int rows() const { return TestSpace::local_dim; }
        constexpr int cols() const { return TestSpace::n_components; }
        constexpr int input_size() const { return StaticInputSize; }
        constexpr int size() const { return rows() * cols(); }
    };
    // fe assembly evaluation
    constexpr Scalar eval(int i, int j, const InputType& fe_packet) const { return fe_packet.test_value(i, j); }
    constexpr Scalar eval(int i, const InputType& fe_packet) const { return eval(i, 0, fe_packet); }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr int input_size() const { return StaticInputSize; }
    constexpr int size() const { return Rows * Cols; }
   private:
    TestSpace* fe_space_;
};

template <typename FeSpace_>
struct scalar_trial_function_impl : public fdapde::ScalarBase<FeSpace_::local_dim, TrialFunction<FeSpace_>> {
    using TrialSpace = std::decay_t<FeSpace_>;
    using Base = fdapde::ScalarBase<FeSpace_::local_dim, TrialFunction<FeSpace_>>;
    using InputType = internals::fe_assembler_packet<TrialSpace::local_dim, TrialSpace::n_components>;
    using Scalar = double;
    static constexpr int StaticInputSize = TrialSpace::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | bilinear_bits::compute_shape_values;

    constexpr scalar_trial_function_impl() = default;
    constexpr scalar_trial_function_impl(FeSpace_& fe_space) : fe_space_(&fe_space) { }

    struct FirstPartialDerivative : fdapde::ScalarBase<TrialSpace::local_dim, FirstPartialDerivative> {
        using Derived = TrialFunction<FeSpace_>;
        using InputType = internals::fe_assembler_packet<TrialSpace::local_dim, TrialSpace::n_components>;
        using Scalar = double;
        static constexpr int StaticInputSize = TrialSpace::local_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | bilinear_bits::compute_shape_grad;

        FirstPartialDerivative() = default;
        FirstPartialDerivative([[maybe_unused]] const Derived& f, int i) : i_(i) { }
        // fe assembly evaluation
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.trial_grad[i_]; }
        constexpr int input_size() const { return StaticInputSize; }
       private:
        int i_;
    };
    // fe assembly evaluation
    constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.trial_value[0]; }
    constexpr TrialSpace& fe_space() { return *fe_space_; }
    constexpr const TrialSpace& fe_space() const { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }  
   private:
    TrialSpace* fe_space_;
};

template <typename FeSpace_>
struct vector_trial_function_impl : public fdapde::MatrixBase<FeSpace_::local_dim, TrialFunction<FeSpace_>> {
    using TrialSpace = std::decay_t<FeSpace_>;
    using Base = fdapde::MatrixBase<FeSpace_::local_dim, TrialFunction<FeSpace_>>;
    using InputType = internals::fe_assembler_packet<TrialSpace::local_dim, TrialSpace::n_components>;
    using Scalar = double;
    static constexpr int StaticInputSize = TrialSpace::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | bilinear_bits::compute_shape_values;
    static constexpr int ReadOnly = 1;
    static constexpr int Rows = FeSpace_::n_components;
    static constexpr int Cols = 1;

    constexpr vector_trial_function_impl() = default;
    constexpr vector_trial_function_impl(FeSpace_& fe_space) : fe_space_(&fe_space) { }

    struct Jacobian : fdapde::MatrixBase<TrialSpace::local_dim, Jacobian> {
        using Derived = TestFunction<FeSpace_>;
        using InputType = internals::fe_assembler_packet<TrialSpace::local_dim, TrialSpace::n_components>;
        using Scalar = double;
        static constexpr int StaticInputSize = TrialSpace::local_dim;
        static constexpr int Rows = TrialSpace::local_dim;
        static constexpr int Cols = TrialSpace::n_components;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | bilinear_bits::compute_shape_grad;

        Jacobian() = default;
        // fe assembly evaluation
        constexpr Scalar eval(int i, int j, const InputType& fe_packet) const { return fe_packet.trial_grad(i, j); }
        constexpr int rows() const { return TrialSpace::local_dim; }
        constexpr int cols() const { return TrialSpace::n_components; }
        constexpr int input_size() const { return StaticInputSize; }
        constexpr int size() const { return rows() * cols(); }
    };
    // fe assembly evaluation
    constexpr Scalar eval(int i, int j, const InputType& fe_packet) const { return fe_packet.trial_value(i, j); }
    constexpr Scalar eval(int i, const InputType& fe_packet) const { return eval(i, 0, fe_packet); }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr int input_size() const { return StaticInputSize; }
    constexpr int size() const { return Rows * Cols; }
   private:
    TrialSpace* fe_space_;
};

}   // namespace internals

template <typename FeSpace_>
struct TestFunction :
    public std::conditional_t<
      FeSpace_::n_components == 1, internals::scalar_test_function_impl<FeSpace_>,
      internals::vector_test_function_impl<FeSpace_>> {
    using Base = std::conditional_t<
      FeSpace_::n_components == 1, internals::scalar_test_function_impl<FeSpace_>,
      internals::vector_test_function_impl<FeSpace_>>;
    constexpr TestFunction() = default;
    constexpr TestFunction(FeSpace_& fe_space) : Base(fe_space) { }
};

// partial derivative of scalar test function
template <typename FeSpace_>
struct PartialDerivative<TestFunction<FeSpace_>, 1> : public TestFunction<FeSpace_>::FirstPartialDerivative {
    PartialDerivative() = default;
    PartialDerivative(const TestFunction<FeSpace_>& f, int i) : TestFunction<FeSpace_>::FirstPartialDerivative(f, i) { }
};
// gradient of vectorial test function (we return directly a custom jacobian implementation)
template <typename FeSpace_>
typename TestFunction<FeSpace_>::Jacobian constexpr grad(const TestFunction<FeSpace_>& xpr)
    requires(FeSpace_::n_components > 1) {
    return typename TestFunction<FeSpace_>::Jacobian();
}

template <typename FeSpace_>
struct TrialFunction : public std::conditional_t<
      FeSpace_::n_components == 1, internals::scalar_trial_function_impl<FeSpace_>,
      internals::vector_trial_function_impl<FeSpace_>> {
    using Base = std::conditional_t<
      FeSpace_::n_components == 1, internals::scalar_trial_function_impl<FeSpace_>,
      internals::vector_trial_function_impl<FeSpace_>>;
    using TrialSpace = typename Base::TrialSpace;
    static constexpr int local_dim = FeSpace_::local_dim;
    static constexpr int embed_dim = FeSpace_::embed_dim;

    constexpr TrialFunction() = default;
    constexpr TrialFunction(FeSpace_& fe_space) : Base(fe_space) { }
    // norm evaluation
    double l2_squared_norm() {
        internals::fe_mass_assembly_loop<DofHandler<local_dim, embed_dim>, typename TrialSpace::FeType> assembler(
          Base::fe_space_->dof_handler());
        return coeff_.dot(assembler.run() * coeff_);
    }
    double l2_norm() { return std::sqrt(l2_squared_norm()); }
    double h1_squared_norm() const {   // Sobolev H^1 norm of finite element function
        TrialFunction u(*Base::fe_space_);
        TestFunction v(*Base::fe_space_);
        auto assembler = integrate(*Base::fe_space_->triangulation())(u * v + inner(grad(u), grad(v)));
        return coeff_.dot(assembler.run() * coeff_);
    }
    double h1_norm() const { return std::sqrt(h1_squared_norm()); }
    const DVector<double>& coeff() const { return coeff_; }
    void set_coeff(const DVector<double>& coeff) { coeff_ = coeff; }
   private:
    DVector<double> coeff_;
};

// partial derivative of scalar trial function
template <typename FeSpace_>
struct PartialDerivative<TrialFunction<FeSpace_>, 1> : public TrialFunction<FeSpace_>::FirstPartialDerivative {
    PartialDerivative() = default;
    PartialDerivative(const TrialFunction<FeSpace_>& f, int i) :
        TrialFunction<FeSpace_>::FirstPartialDerivative(f, i) { }
};
// gradient of vectorial trial function (we return directly a custom jacobian implementation)
template <typename FeSpace_>
typename TrialFunction<FeSpace_>::Jacobian constexpr grad(const TrialFunction<FeSpace_>& xpr)
    requires(FeSpace_::n_components > 1) {
    return typename TrialFunction<FeSpace_>::Jacobian();
}

// representation of u(x) = \sum_{i=1}^{n_dofs} u_i \psi_i(x) with \{ \psi_i \}_i a finite element basis system
template <typename FeSpace_>
class FeFunction :
    public std::conditional_t<
      FeSpace_::n_components == 1, fdapde::ScalarBase<FeSpace_::local_dim, FeFunction<FeSpace_>>,
      fdapde::MatrixBase<FeSpace_::local_dim, FeFunction<FeSpace_>>> {
    using Triangulation = typename FeSpace_::Triangulation;
   public:
    using FeSpace = std::decay_t<FeSpace_>;
    using Base = std::conditional_t<
      FeSpace::n_components == 1, fdapde::ScalarBase<FeSpace::local_dim, FeFunction<FeSpace>>,
      fdapde::MatrixBase<FeSpace::local_dim, FeFunction<FeSpace>>>;
    using InputType = SVector<FeSpace::local_dim>;
    using Scalar = double;
    using OutputType =
      std::conditional_t<FeSpace::n_components == 1, double, Eigen::Matrix<Scalar, FeSpace::n_components, 1>>;
    static constexpr int StaticInputSize = FeSpace::local_dim;
    static constexpr int Rows = FeSpace::n_components == 1 ? 1 : FeSpace::n_components;
    static constexpr int Cols = 1;
    static constexpr int NestAsRef = 1;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int XprBits = 0;

    FeFunction() = default;
    FeFunction(FeSpace_& fe_space) : fe_space_(&fe_space) { coeff_ = DVector<double>::Zero(fe_space_->n_dofs()); }
    FeFunction(FeSpace_& fe_space, const DVector<double>& coeff) : fe_space_(&fe_space), coeff_(coeff) {
        fdapde_assert(coeff.size() > 0 && coeff.size() == fe_space_->n_dofs());
    }
    OutputType operator()(const InputType& p) const {
        int e_id = fe_space_->triangulation().locate(p);
        if (e_id == -1) return std::numeric_limits<Scalar>::quiet_NaN();   // return NaN if point lies outside domain
        // map p to reference cell and evaluate
        typename DofHandler<local_dim, embed_dim>::CellType cell = fe_space_->dof_handler().cell(e_id);
        InputType ref_p = cell.invJ() * (p - cell.node(0));
        DVector<int> active_dofs = cell.dofs();
        OutputType value;
        if constexpr (FeSpace::n_components == 1) {
            value = 0;
        } else {
            value = OutputType::Zero();
        }
        for (int i = 0, n = fe_space_->n_basis(); i < n; ++i) {
            value += coeff_[active_dofs[i]] * fe_space_->eval_shape_function(i, ref_p);
        }
        return value;
    }
    Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(FeSpace::n_components > 1, THIS_METHOD_IS_FOR_VECTOR_FINITE_ELEMENT_FUNCTIONS_ONLY);
        int e_id = fe_space_->triangulation().locate(p);
        if (e_id == -1) return std::numeric_limits<Scalar>::quiet_NaN();   // return NaN if point lies outside domain
        // map p to reference cell and evaluate
        typename DofHandler<local_dim, embed_dim>::CellType cell = fe_space_->dof_handler().cell(e_id);
        InputType ref_p = cell.invJ() * (p - cell.node(0));
        DVector<int> active_dofs = cell.dofs();
        Scalar value = 0;
        for (int j = 0, n = fe_space_->n_basis(); j < n; ++j) {
	  value += coeff_[active_dofs[j]] * fe_space_->eval_shape_function(j, ref_p)[i];
        }
        return value;
    }
    Scalar eval(int i, [[maybe_unused]] int j, const InputType& p) const { return eval(i, p); }

    // i-th component of vector fe_function
    struct VectorFeFunctionComponent : public fdapde::ScalarBase<StaticInputSize, VectorFeFunctionComponent> {
        using Derived = FeFunction<FeSpace_>;
        using InputType = internals::fe_assembler_packet<FeSpace::local_dim, FeSpace::n_components>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpace::local_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | Derived::XprBits;

        VectorFeFunctionComponent() = default;
        VectorFeFunctionComponent(const FeFunction<FeSpace_>* fe_function, int i) : fe_function_(fe_function), i_(i) { }
        Scalar operator()(const InputType& p) const { return fe_function_.eval(i_, p); }
       private:
        const FeFunction<FeSpace_>* fe_function_;
        int i_;
    };
    VectorFeFunctionComponent operator[](int i) const {
        fdapde_static_assert(FeSpace::n_components > 1, THIS_METHOD_IS_FOR_VECTOR_FINITE_ELEMENT_FUNCTIONS_ONLY);
        return VectorFeFunctionComponent(this, i);
    }
    VectorFeFunctionComponent operator()(int i, [[maybe_unused]] int j) const { return operator[](i); }
    // norms of fe functions
    double l2_squared_norm() {
        internals::fe_mass_assembly_loop<DofHandler<local_dim, embed_dim>, typename FeSpace::FeType> assembler(
          fe_space_->dof_handler());
        return coeff_.dot(assembler.assemble() * coeff_);
    }
    double l2_norm() { return std::sqrt(l2_squared_norm()); }
    double h1_squared_norm() const {   // Sobolev H^1 norm of finite element function
        TrialFunction u(*fe_space_);
        TestFunction  v(*fe_space_);
        auto a = integrate(fe_space_->triangulation())(inner(grad(u), grad(v)) + u * v);
        return coeff_.dot(a.assemble() * coeff_);
    }
    double h1_norm() const { return std::sqrt(h1_squared_norm()); }

    // getters
    const DVector<double>& coeff() const { return coeff_; }
    constexpr FeSpace& fe_space() { return *fe_space_; }
    constexpr const FeSpace& fe_space() const { return *fe_space_; }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr int input_size() const { return StaticInputSize; }
    void set_coeff(const DVector<double>& coeff) { coeff_ = coeff; }
    // linear algebra between fe functions
    friend constexpr FeFunction<FeSpace_> operator+(FeFunction<FeSpace_>& lhs, FeFunction<FeSpace_>& rhs) {
        return FeFunction<FeSpace_>(lhs.fe_space(), lhs.coeff() + rhs.coeff());
    }
    friend constexpr FeFunction<FeSpace_> operator-(FeFunction<FeSpace_>& lhs, FeFunction<FeSpace_>& rhs) {
        return FeFunction<FeSpace_>(lhs.fe_space(), lhs.coeff() - rhs.coeff());
    }
    // assignment from expansion coefficient vector
    FeFunction& operator=(const DVector<double>& coeff) {
        fdapde_assert(coeff.size() > 0 && coeff.size() == fe_space_->n_dofs());
        coeff_ = coeff;
        return *this;
    }
    // static constructors
    static constexpr FeFunction<FeSpace_> Zero(FeSpace_& fe_space) {
        return FeFunction<FeSpace_>(fe_space, DVector<Scalar>::Zero(fe_space.n_dofs()));
    }
   private:
    DVector<double> coeff_;
    FeSpace* fe_space_;
};

}   // namespace fdapde

#endif   // __FE_FUNCTION_H__
