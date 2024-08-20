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

template <typename FeSpace_>
struct TestFunction : public fdapde::ScalarBase<FeSpace_::local_dim, TestFunction<FeSpace_>> {
    using TestSpace = std::decay_t<FeSpace_>;
    using InputType = internals::fe_assembler_packet<TestSpace::local_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = TestSpace::local_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | bilinear_bits::compute_shape_values;
  
    constexpr TestFunction() = default;
    constexpr TestFunction(FeSpace_& fe_space) : fe_space_(&fe_space) { }
    constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_value; }
    constexpr TestSpace& fe_space() { return *fe_space_; }
    constexpr const TestSpace& fe_space() const { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }

    struct FirstPartialDerivative : fdapde::ScalarBase<TestSpace::local_dim, FirstPartialDerivative> {
        using Derived = TestFunction<FeSpace_>;
        using InputType = internals::fe_assembler_packet<TestSpace::local_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = TestSpace::local_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | bilinear_bits::compute_shape_grad;

        FirstPartialDerivative() = default;
        FirstPartialDerivative(const Derived& f, int i) : f_(f), i_(i) { }
        // fe assembly evaluation
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_grad[i_]; }
        constexpr int input_size() const { return f_.input_size(); }
       private:
        typename internals::ref_select<Derived>::type f_;
        int i_;
    };
   private:
    TestSpace* fe_space_;
};

template <typename FeSpace_>
struct PartialDerivative<TestFunction<FeSpace_>, 1> : public TestFunction<FeSpace_>::FirstPartialDerivative {
    PartialDerivative() = default;
    PartialDerivative(const TestFunction<FeSpace_>& f, int i) : TestFunction<FeSpace_>::FirstPartialDerivative(f, i) { }
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

    // norm evaluation
    double l2_squared_norm() {
        internals::fe_assembler_mass_loop<DofHandler<local_dim, embed_dim>, typename TrialSpace::FeType> assembler(
          fe_space_->dof_handler());
        return coeff_.dot(assembler.run() * coeff_);
    }
    double l2_norm() { return std::sqrt(l2_squared_norm()); }
    double h1_squared_norm() const {   // Sobolev H^1 norm of finite element function
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
    constexpr const TrialSpace& fe_space() const { return *fe_space_; }
    const DVector<double>& coeff() const { return coeff_; }
    constexpr int input_size() const { return StaticInputSize; }
    void set_coeff(const DVector<double>& coeff) { coeff_ = coeff; }

    struct FirstPartialDerivative : fdapde::ScalarBase<TrialSpace::local_dim, FirstPartialDerivative> {
        using Derived = TrialFunction<FeSpace_>;
        using InputType = internals::fe_assembler_packet<TrialSpace::local_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = TrialSpace::local_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | bilinear_bits::compute_shape_grad;

        FirstPartialDerivative() = default;
        FirstPartialDerivative(const Derived& f, int i) : f_(f), i_(i) { }
        // fe assembly evaluation
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.trial_grad[i_]; }
        constexpr int input_size() const { return f_.input_size(); }
       private:
        typename internals::ref_select<Derived>::type f_;
        int i_;
    };
   private:
    DVector<double> coeff_;
    TrialSpace* fe_space_;
};

template <typename FeSpace_>
struct PartialDerivative<TrialFunction<FeSpace_>, 1> : public TrialFunction<FeSpace_>::FirstPartialDerivative {
    PartialDerivative() = default;
    PartialDerivative(const TrialFunction<FeSpace_>& f, int i) :
        TrialFunction<FeSpace_>::FirstPartialDerivative(f, i) { }
};

// representation of u(x) = \sum_{i=1}^{n_dofs} u_i \psi_i(x) with \{ \psi_i \}_i a finite element basis
template <typename FeSpace_>
class FeFunction : public fdapde::ScalarBase<FeSpace_::local_dim, FeFunction<FeSpace_>> {
    using Triangulation = typename FeSpace_::Triangulation;
   public:
    using FeSpace = std::decay_t<FeSpace_>;
    using Base = ScalarBase<FeSpace::local_dim, FeFunction<FeSpace>>;
    using InputType = SVector<FeSpace::local_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = FeSpace::local_dim;
    static constexpr int NestAsRef = 1;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int XprBits = 0;

    FeFunction() = default;
    FeFunction(FeSpace_& fe_space) : fe_space_(&fe_space) { coeff_ = DVector<double>::Zero(fe_space_->n_dofs()); }
    FeFunction(FeSpace_& fe_space, const DVector<double>& coeff) : fe_space_(&fe_space), coeff_(coeff) {
        fdapde_assert(coeff.size() > 0 && coeff.size() == fe_space_->n_dofs());
    }
    constexpr Scalar operator()(const InputType& p) const {
        int e_id = fe_space_->triangulation().locate(p);
        if (e_id == -1) return std::numeric_limits<Scalar>::quiet_NaN();   // return NaN if point lies outside domain
        // map p to reference cell and evaluate
        typename DofHandler<local_dim, embed_dim>::CellType cell = fe_space_->dof_handler().cell(e_id);
        InputType ref_p = cell.invJ() * (p - cell.node(0));
	DVector<int> active_dofs = cell.dofs();
        Scalar value = 0;
        for (int i = 0; i < fe_space_->n_basis(); ++i) { value += coeff_[active_dofs[i]] * fe_space_->eval(i, ref_p); }
        return value;
    }
    // norms of fe functions
    double l2_squared_norm() {
        internals::fe_assembler_mass_loop<DofHandler<local_dim, embed_dim>, typename FeSpace::FeType> assembler(
          fe_space_->dof_handler());
        return coeff_.dot(assembler.run() * coeff_);
    }
    double l2_norm() { return std::sqrt(l2_squared_norm()); }
    double h1_squared_norm() const {   // Sobolev H^1 norm of finite element function
        TrialFunction u(*fe_space_);
        TestFunction  v(*fe_space_);
        auto assembler = integrate(fe_space_->triangulation())(u * v + inner(grad(u), grad(v)));
        return coeff_.dot(assembler.run() * coeff_);
    }
    double h1_norm() const { return std::sqrt(h1_squared_norm()); }

    // getters
    const DVector<double>& coeff() const { return coeff_; }
    constexpr FeSpace& fe_space() { return *fe_space_; }
    constexpr const FeSpace& fe_space() const { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }
    void set_coeff(const DVector<double>& coeff) { coeff_ = coeff; }

    // linear algebra between FeFunctions
    friend constexpr FeFunction<FeSpace_> operator+(FeFunction<FeSpace_>& lhs, FeFunction<FeSpace_>& rhs) {
        return FeFunction<FeSpace_>(lhs.fe_space(), lhs.coeff() + rhs.coeff());
    }
    friend constexpr FeFunction<FeSpace_> operator-(FeFunction<FeSpace_>& lhs, FeFunction<FeSpace_>& rhs) {
        return FeFunction<FeSpace_>(lhs.fe_space(), lhs.coeff() - rhs.coeff());
    }
   private:
    DVector<double> coeff_;
    FeSpace* fe_space_;
};

}   // namespace fdapde

#endif   // __FE_FUNCTION_H__
