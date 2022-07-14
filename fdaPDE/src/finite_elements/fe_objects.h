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

#ifndef __FDAPDE_FE_OBJECTS_H__
#define __FDAPDE_FE_OBJECTS_H__

#include "header_check.h"

namespace fdapde {  
namespace internals {

// scalar finite element support
template <typename FeSpace_>
struct fe_scalar_test_function_impl :
    public ScalarFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, TestFunction<FeSpace_, finite_element_tag>> {
   private:
    using pointer_t = std::conditional_t<
      is_shared_ptr_v<std::decay_t<FeSpace_>>, std::decay_t<FeSpace_>, const std::decay_t<FeSpace_>*>;
   public:
    using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
    // ScalarFieldBase interface
    using Base = ScalarFieldBase<FeSpaceClean::embed_dim, TestFunction<FeSpace_, finite_element_tag>>;
    using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_values);
    // assembly interace
    using TestSpace = std::decay_t<FeSpaceClean>;

    constexpr fe_scalar_test_function_impl() noexcept = default;
    constexpr fe_scalar_test_function_impl(const FeSpace_& fe_space) noexcept : fe_space_() {
        if constexpr (is_shared_ptr_v<std::decay_t<FeSpace_>>) {
            fe_space_ = fe_space;
        } else {
            fe_space_ = std::addressof(fe_space);
        }
    }
    constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_value(0); }

    // first partial derivative functor
    template <typename Derived_>
    struct FirstPartialDerivative :
        ScalarFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, FirstPartialDerivative<Derived_>> {
        using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
        using Base = ScalarFieldBase<FeSpaceClean::embed_dim, FirstPartialDerivative<Derived_>>;
        using Derived = Derived_;
        template <typename T> using Meta = FirstPartialDerivative<T>;
        using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_grad);
        using TestSpace = std::decay_t<FeSpaceClean>;

        FirstPartialDerivative() noexcept = default;
        FirstPartialDerivative(const Derived_& xpr) noexcept : Base(), xpr_(xpr), i_(0) { }
        FirstPartialDerivative(const Derived_& xpr, int i) noexcept : Base(), xpr_(xpr), i_(i) { }
        // accessors
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
        constexpr const TestSpace& function_space() const { return xpr_.function_space(); }
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_grad[i_]; }
       protected:
        int i_;
        Derived xpr_;
    };
    // second mixed partial derivative functor
    template <typename Derived_>
    struct MixedPartialDerivative :
        ScalarFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, MixedPartialDerivative<Derived_>> {
        using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
        using Base = ScalarFieldBase<FeSpaceClean::embed_dim, MixedPartialDerivative<Derived_>>;
        using Derived = Derived_;
        template <typename T> using Meta = MixedPartialDerivative<T>;
        using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_hess);
        using TestSpace = std::decay_t<FeSpaceClean>;

        MixedPartialDerivative() noexcept = default;
        MixedPartialDerivative(const Derived_& xpr) noexcept : Base(), xpr_(xpr), i_(0), j_(0) { }
        MixedPartialDerivative(const Derived_& xpr, int i, int j) noexcept : Base(), xpr_(xpr), i_(i), j_(j) { }
        // accessors
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
        constexpr const TestSpace& function_space() const { return xpr_.function_space(); }
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_hess(0, i_, j_); }
       protected:
        int i_, j_;
        Derived xpr_;
    };
    // accessors
    constexpr const TestSpace& function_space() const { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }
   protected:
    pointer_t fe_space_;
};

template <typename FeSpace_>
struct fe_scalar_trial_function_impl :
    public ScalarFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, TrialFunction<FeSpace_, finite_element_tag>> {
   private:
    using pointer_t = std::conditional_t<
      is_shared_ptr_v<std::decay_t<FeSpace_>>, std::decay_t<FeSpace_>, const std::decay_t<FeSpace_>*>;
   public:
    using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
    // ScalarFieldBase interface
    using Base = ScalarFieldBase<FeSpaceClean::embed_dim, TrialFunction<FeSpace_, finite_element_tag>>;
    using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_values);
    // assembly interace
    using TrialSpace = std::decay_t<FeSpaceClean>;

    constexpr fe_scalar_trial_function_impl() noexcept = default;
    constexpr fe_scalar_trial_function_impl(const FeSpace_& fe_space) noexcept : fe_space_() {
        if constexpr (is_shared_ptr_v<std::decay_t<FeSpace_>>) {
            fe_space_ = fe_space;
        } else {
            fe_space_ = std::addressof(fe_space);
        }
    }
    constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.trial_value(0); }

    // first partial derivative functor
    template <typename Derived_>
    struct FirstPartialDerivative :
        ScalarFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, FirstPartialDerivative<Derived_>> {
        using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
        using Base = ScalarFieldBase<FeSpaceClean::embed_dim, FirstPartialDerivative<Derived_>>;
        using Derived = Derived_;
        template <typename T> using Meta = FirstPartialDerivative<T>;
        using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_grad);
        using TrialSpace = std::decay_t<FeSpaceClean>;

        FirstPartialDerivative() noexcept = default;
        FirstPartialDerivative(const Derived_& xpr) noexcept : Base(), xpr_(xpr), i_(0) { }
        FirstPartialDerivative(const Derived_& xpr, int i) noexcept : Base(), xpr_(xpr), i_(i) { }
        // accessors
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
        constexpr const TrialSpace& function_space() const { return xpr_.function_space(); }
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.trial_grad[i_]; }
       protected:
        int i_;
        Derived xpr_;
    };
    // second mixed partial derivative functor
    template <typename Derived_>
    struct MixedPartialDerivative :
        ScalarFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, MixedPartialDerivative<Derived_>> {
        using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
        using Base = ScalarFieldBase<FeSpaceClean::embed_dim, MixedPartialDerivative<Derived_>>;
        using Derived = Derived_;
        template <typename T> using Meta = MixedPartialDerivative<T>;
        using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_hess);
        using TrialSpace = std::decay_t<FeSpaceClean>;

        MixedPartialDerivative() noexcept = default;
        MixedPartialDerivative(const Derived_& xpr) noexcept : Base(), xpr_(xpr), i_(0), j_(0) { }
        MixedPartialDerivative(const Derived_& xpr, int i, int j) noexcept : Base(), xpr_(xpr), i_(i), j_(j) { }
        // accessors
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
        constexpr const TrialSpace& function_space() const { return xpr_.function_space(); }
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.trial_hess(0, i_, j_); }
       protected:
        int i_, j_;
        Derived xpr_;
    };
    // accessors
    constexpr const TrialSpace& function_space() const { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }
   protected:
    pointer_t fe_space_;
};

// vector finite element support
template <typename FeSpace_>
struct fe_vector_test_function_impl :
    public MatrixFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, TestFunction<FeSpace_, finite_element_tag>> {
   private:
    using pointer_t = std::conditional_t<
      is_shared_ptr_v<std::decay_t<FeSpace_>>, std::decay_t<FeSpace_>, const std::decay_t<FeSpace_>*>;
   public:
    using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
    // MatrixFieldBase interface
    using Base = MatrixFieldBase<FeSpaceClean::embed_dim, TestFunction<FeSpace_, finite_element_tag>>;
    using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_values);
    static constexpr int ReadOnly = 1;
    static constexpr int Rows = FeSpaceClean::n_components;
    static constexpr int Cols = 1;
    // assembly interface
    using TestSpace = std::decay_t<FeSpaceClean>;

    constexpr fe_vector_test_function_impl() noexcept = default;
    constexpr fe_vector_test_function_impl(const FeSpace_& fe_space) noexcept : fe_space_() {
        if constexpr (is_shared_ptr_v<std::decay_t<FeSpace_>>) {
            fe_space_ = fe_space;
        } else {
            fe_space_ = std::addressof(fe_space);
        }
    }
    constexpr Scalar eval(int i, const typename Base::InputType& fe_packet) const { return fe_packet.test_value(i); }
    constexpr Scalar eval(int i, [[maybe_unused]] int j, const typename Base::InputType& fe_packet) const {
        return fe_packet.test_value(i);
    }

    template <typename Derived__>
    struct Jacobian : MatrixFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, Jacobian<Derived__>> {
        using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
        using Base = MatrixFieldBase<FeSpaceClean::embed_dim, Jacobian<Derived__>>;
        using Derived = Derived__;
        template <typename T> using Meta = Jacobian<T>;
        using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_grad);
        static constexpr int ReadOnly = 1;
        static constexpr int Rows = FeSpaceClean::local_dim;
        static constexpr int Cols = FeSpaceClean::n_components;
        using TestSpace = std::decay_t<FeSpaceClean>;

        explicit constexpr Jacobian(const Derived__& xpr) noexcept : Base(), xpr_(xpr) { }
        // accessors
        constexpr int rows() const { return Rows; }
        constexpr int cols() const { return Cols; }
        constexpr int input_size() const { return StaticInputSize; }
        constexpr int size() const { return Rows * Cols; }
        constexpr const Derived& derived() const { return xpr_; }
        constexpr const TestSpace& function_space() const { return xpr_.function_space(); }
        constexpr Scalar eval(int i, int j, const InputType& fe_packet) const { return fe_packet.test_grad(i, j); }
       protected:
        Derived xpr_;
    };
    template <typename Derived__>
    struct Divergence : ScalarFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, Divergence<Derived__>> {
        using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
        using Base = ScalarFieldBase<FeSpaceClean::embed_dim, Divergence<Derived__>>;
        using Derived = Derived__;
        template <typename T> using Meta = Divergence<T>;
        using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_div);
        using TestSpace = std::decay_t<FeSpaceClean>;

        explicit constexpr Divergence(const Derived__& xpr) noexcept : Base(), xpr_(xpr) { }
        // accessors
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
        constexpr const TestSpace& function_space() const { return xpr_.function_space(); }
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.test_div; }
       protected:
        Derived xpr_;
    };
    // accessors
    constexpr const TestSpace& function_space() const { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }
   protected:
    pointer_t fe_space_;
};

template <typename FeSpace_>
struct fe_vector_trial_function_impl :
    public MatrixFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, TrialFunction<FeSpace_, finite_element_tag>> {
   private:
    using pointer_t = std::conditional_t<
      is_shared_ptr_v<std::decay_t<FeSpace_>>, std::decay_t<FeSpace_>, const std::decay_t<FeSpace_>*>;
   public:
    using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;  
    // MatrixFieldBase interface
    using Base = MatrixFieldBase<FeSpaceClean::embed_dim, TrialFunction<FeSpace_, finite_element_tag>>;
    using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_values);
    static constexpr int ReadOnly = 1;
    static constexpr int Rows = FeSpaceClean::n_components;
    static constexpr int Cols = 1;
    // assembly interface
    using TrialSpace = std::decay_t<FeSpaceClean>;

    constexpr fe_vector_trial_function_impl() noexcept = default;
    constexpr fe_vector_trial_function_impl(const FeSpace_& fe_space) noexcept : fe_space_() {
        if constexpr (is_shared_ptr_v<std::decay_t<FeSpace_>>) {
            fe_space_ = fe_space;
        } else {
            fe_space_ = std::addressof(fe_space);
        }
    }
    constexpr Scalar eval(int i, const typename Base::InputType& fe_packet) const { return fe_packet.trial_value(i); }
    constexpr Scalar eval(int i, [[maybe_unused]] int j, const typename Base::InputType& fe_packet) const {
        return fe_packet.trial_value(i);
    }

    template <typename Derived__>
    struct Jacobian : MatrixFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, Jacobian<Derived__>> {
        using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
        using Base = MatrixFieldBase<FeSpaceClean::embed_dim, Jacobian<Derived__>>;
        using Derived = Derived__;
        template <typename T> using Meta = Jacobian<T>;
        using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_grad);
        static constexpr int ReadOnly = 1;
        static constexpr int Rows = FeSpaceClean::local_dim;
        static constexpr int Cols = FeSpaceClean::n_components;
        using TrialSpace = std::decay_t<FeSpaceClean>;

        explicit constexpr Jacobian(const Derived__& xpr) noexcept : Base(), xpr_(xpr) { }
        // accessors
        constexpr int rows() const { return Rows; }
        constexpr int cols() const { return Cols; }
        constexpr int input_size() const { return StaticInputSize; }
        constexpr int size() const { return Rows * Cols; }
        constexpr const Derived& derived() const { return xpr_; }
        constexpr const TrialSpace& function_space() const { return xpr_.function_space(); }
        constexpr Scalar eval(int i, int j, const InputType& fe_packet) const { return fe_packet.trial_grad(i, j); }
       protected:
        Derived xpr_;
    };
    template <typename Derived__>
    struct Divergence : ScalarFieldBase<remove_shared_ptr_t<FeSpace_>::embed_dim, Divergence<Derived__>> {
        using FeSpaceClean = remove_shared_ptr_t<FeSpace_>;
        using Base = ScalarFieldBase<FeSpaceClean::embed_dim, Divergence<Derived__>>;
        using Derived = Derived__;
        template <typename T> using Meta = Divergence<T>;
        using InputType = fe_assembler_packet<FeSpaceClean::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpaceClean::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(fe_assembler_flags::compute_shape_div);
        using TrialSpace = std::decay_t<FeSpaceClean>;

        explicit constexpr Divergence(const Derived__& xpr) noexcept : Base(), xpr_(xpr) { }
        // accessors
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
        constexpr const TrialSpace& function_space() const { return xpr_.function_space(); }
        constexpr Scalar operator()(const InputType& fe_packet) const { return fe_packet.trial_div; }
       protected:
        Derived xpr_;
    };
    // accessors
    constexpr const TrialSpace& function_space() const { return *fe_space_; }
    constexpr int input_size() const { return StaticInputSize; }
   protected:
    pointer_t fe_space_;
};

}   // namespace internals

// public test function type
template <typename FeSpace_>
    requires(
      (internals::is_shared_ptr_v<FeSpace_> &&
       std::is_same_v<
         typename internals::remove_shared_ptr_t<FeSpace_>::discretization_category, finite_element_tag>) ||
      std::is_same_v<typename std::decay_t<FeSpace_>::discretization_category, finite_element_tag>)
struct TestFunction<FeSpace_, finite_element_tag> :
    public std::conditional_t<
      internals::remove_shared_ptr_t<FeSpace_>::n_components == 1, internals::fe_scalar_test_function_impl<FeSpace_>,
      internals::fe_vector_test_function_impl<FeSpace_>> {
    using Base = std::conditional_t<
      internals::remove_shared_ptr_t<FeSpace_>::n_components == 1, internals::fe_scalar_test_function_impl<FeSpace_>,
      internals::fe_vector_test_function_impl<FeSpace_>>;
    constexpr TestFunction() = default;
    constexpr TestFunction(const FeSpace_& fe_space) : Base(fe_space) { }
};
// partial derivatives of scalar test function
template <typename FeSpace_>
struct PartialDerivative<TestFunction<FeSpace_, finite_element_tag>, 1> :
    public TestFunction<FeSpace_, finite_element_tag>::template FirstPartialDerivative<
      TestFunction<FeSpace_, finite_element_tag>> {
    using Base = TestFunction<FeSpace_, finite_element_tag>::template FirstPartialDerivative<
      TestFunction<FeSpace_, finite_element_tag>>;
    PartialDerivative() = default;
    PartialDerivative(const TestFunction<FeSpace_, finite_element_tag>& xpr, int i) : Base(xpr, i) { }
};
template <typename FeSpace_>
struct PartialDerivative<TestFunction<FeSpace_, finite_element_tag>, 2> :
    public TestFunction<FeSpace_, finite_element_tag>::template MixedPartialDerivative<
      TestFunction<FeSpace_, finite_element_tag>> {
    using Base = TestFunction<FeSpace_, finite_element_tag>::template MixedPartialDerivative<
      TestFunction<FeSpace_, finite_element_tag>>;
    PartialDerivative() = default;
    PartialDerivative(const TestFunction<FeSpace_, finite_element_tag>& xpr, int i, int j) : Base(xpr, i, j) { }
};
// gradient of vectorial test function
template <typename FeSpace_>
    requires(internals::remove_shared_ptr_t<FeSpace_>::n_components > 1)
constexpr auto grad(const TestFunction<FeSpace_, finite_element_tag>& xpr) {
    return typename TestFunction<FeSpace_, finite_element_tag>::template Jacobian<
      TestFunction<FeSpace_, finite_element_tag>>(xpr);
}
// divergence of vectorial test function
template <typename FeSpace_>
    requires(internals::remove_shared_ptr_t<FeSpace_>::n_components > 1)
constexpr auto div(const TestFunction<FeSpace_, finite_element_tag>& xpr) {
    return typename TestFunction<FeSpace_, finite_element_tag>::template Divergence<
      TestFunction<FeSpace_, finite_element_tag>>(xpr);
}

// public trial function type
template <typename FeSpace_>
    requires(
      (internals::is_shared_ptr_v<FeSpace_> &&
       std::is_same_v<
         typename internals::remove_shared_ptr_t<FeSpace_>::discretization_category, finite_element_tag>) ||
      std::is_same_v<typename std::decay_t<FeSpace_>::discretization_category, finite_element_tag>)
struct TrialFunction<FeSpace_, finite_element_tag> :
    public std::conditional_t<
      internals::remove_shared_ptr_t<FeSpace_>::n_components == 1, internals::fe_scalar_trial_function_impl<FeSpace_>,
      internals::fe_vector_trial_function_impl<FeSpace_>> {
    using Base = std::conditional_t<
      internals::remove_shared_ptr_t<FeSpace_>::n_components == 1, internals::fe_scalar_trial_function_impl<FeSpace_>,
      internals::fe_vector_trial_function_impl<FeSpace_>>;
    using FeSpaceClean = internals::remove_shared_ptr_t<FeSpace_>;
    using TrialSpace = typename Base::TrialSpace;
    static constexpr int local_dim = FeSpaceClean::local_dim;
    static constexpr int embed_dim = FeSpaceClean::embed_dim;
  
    constexpr TrialFunction() = default;
    constexpr TrialFunction(const FeSpace_& fe_space) : Base(fe_space) { }
    // norm evaluation
    double l2_squared_norm() {
        internals::fe_mass_assembly_loop<typename TrialSpace::FeType> assembler(Base::fe_space_->dof_handler());
        return coeff_.dot(assembler.assemble() * coeff_);
    }
    double l2_norm() { return std::sqrt(l2_squared_norm()); }
    double h1_squared_norm() const {   // Sobolev H^1 norm of finite element function
        TrialFunction u(*Base::fe_space_);
        TestFunction  v(*Base::fe_space_);
        auto assembler = integrate(*Base::fe_space_->triangulation())(u * v + inner(grad(u), grad(v)));
        return coeff_.dot(assembler.assemble() * coeff_);
    }
    double h1_norm() const { return std::sqrt(h1_squared_norm()); }
    const Eigen::Matrix<double, Dynamic, 1>& coeff() const { return coeff_; }
    void set_coeff(const Eigen::Matrix<double, Dynamic, 1>& coeff) { coeff_ = coeff; }
   private:
    Eigen::Matrix<double, Dynamic, 1> coeff_;
};
// partial derivative of scalar trial function
template <typename FeSpace_>
struct PartialDerivative<TrialFunction<FeSpace_, finite_element_tag>, 1> :
    public TrialFunction<FeSpace_, finite_element_tag>::template FirstPartialDerivative<
      TrialFunction<FeSpace_, finite_element_tag>> {
    using Base = TrialFunction<FeSpace_, finite_element_tag>::template FirstPartialDerivative<
      TrialFunction<FeSpace_, finite_element_tag>>;
    PartialDerivative() = default;
    PartialDerivative(const TrialFunction<FeSpace_, finite_element_tag>& xpr, int i) : Base(xpr, i) { }
};
template <typename FeSpace_>
struct PartialDerivative<TrialFunction<FeSpace_, finite_element_tag>, 2> :
    public TrialFunction<FeSpace_, finite_element_tag>::template MixedPartialDerivative<
      TrialFunction<FeSpace_, finite_element_tag>> {
    using Base = TrialFunction<FeSpace_, finite_element_tag>::template MixedPartialDerivative<
      TrialFunction<FeSpace_, finite_element_tag>>;
    PartialDerivative() = default;
    PartialDerivative(const TrialFunction<FeSpace_, finite_element_tag>& xpr, int i, int j) : Base(xpr, i, j) { }
};
// gradient of vectorial trial function
template <typename FeSpace_>
    requires(FeSpace_::n_components > 1)
constexpr auto grad(const TrialFunction<FeSpace_, finite_element_tag>& xpr) {
    return typename TrialFunction<FeSpace_, finite_element_tag>::template Jacobian<
      TrialFunction<FeSpace_, finite_element_tag>>(xpr);
}
// divergence of vectorial trial function
template <typename FeSpace_>
    requires(FeSpace_::n_components > 1)
constexpr auto div(const TrialFunction<FeSpace_, finite_element_tag>& xpr) {
    return typename TrialFunction<FeSpace_, finite_element_tag>::template Divergence<
      TrialFunction<FeSpace_, finite_element_tag>>(xpr);
}

// representation of u(x) = \sum_{i=1}^{n_dofs} u_i \psi_i(x) with \{ \psi_i \}_i a finite element basis system
template <typename FeSpace_>
class FeFunction :
    public std::conditional_t<
      internals::remove_shared_ptr_t<FeSpace_>::n_components == 1,
      ScalarFieldBase<internals::remove_shared_ptr_t<FeSpace_>::embed_dim, FeFunction<FeSpace_>>,
      MatrixFieldBase<internals::remove_shared_ptr_t<FeSpace_>::embed_dim, FeFunction<FeSpace_>>> {
    using Triangulation = typename internals::remove_shared_ptr_t<std::decay_t<FeSpace_>>::Triangulation;
   private:
    using pointer_t = std::conditional_t<
      internals::is_shared_ptr_v<std::decay_t<FeSpace_>>, std::decay_t<FeSpace_>, const std::decay_t<FeSpace_>*>;
   public:
    using FeSpace = internals::remove_shared_ptr_t<std::decay_t<FeSpace_>>;
    using Base = std::conditional_t<
      FeSpace::n_components == 1, ScalarFieldBase<FeSpace::embed_dim, FeFunction<FeSpace>>,
      MatrixFieldBase<FeSpace::embed_dim, FeFunction<FeSpace>>>;
    using DofHandlerType = typename FeSpace::DofHandlerType;
    using InputType = Eigen::Matrix<double, FeSpace::embed_dim, 1>;
    using Scalar = double;
    using OutputType =
      std::conditional_t<FeSpace::n_components == 1, double, Eigen::Matrix<Scalar, FeSpace::n_components, 1>>;
    static constexpr int StaticInputSize = FeSpace::embed_dim;
    static constexpr int Rows = FeSpace::n_components == 1 ? 1 : FeSpace::n_components;
    static constexpr int Cols = 1;
    static constexpr int NestAsRef = 1;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int XprBits = 0;

    FeFunction() = default;
    explicit FeFunction(FeSpace_& fe_space) : fe_space_() {
        if constexpr (internals::is_shared_ptr_v<std::decay_t<FeSpace_>>) {
            fe_space_ = fe_space;
        } else {
            fe_space_ = std::addressof(fe_space);
        }
        coeff_ = Eigen::Matrix<double, Dynamic, 1>::Zero(fe_space_->n_dofs());
    }
    FeFunction(FeSpace_& fe_space, const Eigen::Matrix<double, Dynamic, 1>& coeff) : fe_space_(), coeff_(coeff) {
        if constexpr (internals::is_shared_ptr_v<std::decay_t<FeSpace_>>) {
            fe_space_ = fe_space;
        } else {
            fe_space_ = std::addressof(fe_space);
        }
        fdapde_assert(coeff.size() > 0 && coeff.size() == fe_space_->n_dofs());
    }
    OutputType operator()(const InputType& p) const {
        int e_id = fe_space_->triangulation().locate(p);
        if (e_id == -1) return std::numeric_limits<Scalar>::quiet_NaN();   // return NaN if point lies outside domain
        // map p to reference cell and evaluate
        typename DofHandlerType::CellType cell = fe_space_->dof_handler().cell(e_id);
        InputType ref_p = cell.invJ() * (p - cell.node(0));
        Eigen::Matrix<int, Dynamic, 1> active_dofs = cell.dofs();
        OutputType value;
        if constexpr (FeSpace::n_components == 1) {
            value = 0;
        } else {
            value = OutputType::Zero();
        }
        for (int i = 0, n = fe_space_->n_shape_functions(); i < n; ++i) {
            value += coeff_[active_dofs[i]] * fe_space_->eval_shape_value(i, ref_p);
        }
        return value;
    }
    Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(FeSpace::n_components > 1, THIS_METHOD_IS_FOR_VECTOR_FINITE_ELEMENT_FUNCTIONS_ONLY);
        int e_id = fe_space_->triangulation().locate(p);
        if (e_id == -1) return std::numeric_limits<Scalar>::quiet_NaN();   // return NaN if point lies outside domain
        // map p to reference cell and evaluate
        typename DofHandlerType::CellType cell = fe_space_->dof_handler().cell(e_id);
        InputType ref_p = cell.invJ() * (p - cell.node(0));
        Eigen::Matrix<int, Dynamic, 1> active_dofs = cell.dofs();
        Scalar value = 0;
        for (int j = 0, n = fe_space_->n_shape_functions(); j < n; ++j) {
            value += coeff_[active_dofs[j]] * fe_space_->eval_shape_value(j, ref_p)[i];
        }
        return value;
    }
    Scalar eval(int i, [[maybe_unused]] int j, const InputType& p) const { return eval(i, p); }

    // i-th component of vector fe_function
    struct VectorFeFunctionComponent : public ScalarFieldBase<StaticInputSize, VectorFeFunctionComponent> {
        using Derived = FeFunction<FeSpace_>;
        using InputType = internals::fe_assembler_packet<FeSpace::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = FeSpace::embed_dim;
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
    double l2_squared_norm() const {
        internals::fe_mass_assembly_loop<FeSpace> assembler(*fe_space_);
        return coeff_.dot(assembler.assemble() * coeff_);
    }
    double l2_norm() const { return std::sqrt(l2_squared_norm()); }
    double h1_squared_norm() const {   // Sobolev H^1 norm of finite element function
        TrialFunction u(*fe_space_);
        TestFunction  v(*fe_space_);
        auto a = integral(fe_space_->triangulation())(dot(grad(u), grad(v)) + u * v);
        return coeff_.dot(a.assemble() * coeff_);
    }
    double h1_norm() const { return std::sqrt(h1_squared_norm()); }
    // specialized integration routine for fe functions on cell range
    template <typename Triangulation>
    double integrate_on(CellIterator<Triangulation> begin, CellIterator<Triangulation> end) const {
        fdapde_static_assert(FeSpace::n_components == 1, THIS_METHOD_IS_FOR_SCALAR_FINITE_ELEMENT_FUNCTIONS_ONLY);
        using FeType = typename FeSpace::FeType;
        using Quadrature = typename FeType::template cell_quadrature_t<FeSpace::local_dim>;
        using Iterator = CellIterator<Triangulation>;
        constexpr int n_quadrature_nodes = Quadrature::order;

        Quadrature quadrature {};
        Eigen::Map<const Eigen::Matrix<double, n_quadrature_nodes, Triangulation::local_dim, Eigen::RowMajor>>
          ref_quad_nodes(quadrature.nodes.data());
        double integral = 0;
        for (Iterator it = begin; it != end; ++it) {
            double partial = 0;
            typename DofHandlerType::CellType cell = fe_space_->dof_handler().cell(it->id());
            Eigen::Matrix<int, Dynamic, 1> active_dofs = cell.dofs();
	    // integral approximation on cell (skip point location)
            for (int i = 0, n = fe_space_->n_shape_functions(); i < n; ++i) {
                for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                    partial +=
                      (coeff_[active_dofs[i]] * fe_space_->eval_shape_value(i, ref_quad_nodes.row(q_k).transpose())) *
                      quadrature.weights[q_k];
                }
            }
            integral += partial * it->measure();
        }
        return integral;
    }

    // getters
    const Eigen::Matrix<double, Dynamic, 1>& coeff() const { return coeff_; }
    constexpr FeSpace& function_space() { return *fe_space_; }
    constexpr const FeSpace& function_space() const { return *fe_space_; }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr int input_size() const { return StaticInputSize; }
    void set_coeff(const Eigen::Matrix<double, Dynamic, 1>& coeff) { coeff_ = coeff; }
    // linear algebra between fe functions
    friend constexpr FeFunction<FeSpace_> operator+(FeFunction<FeSpace_>& lhs, FeFunction<FeSpace_>& rhs) {
        return FeFunction<FeSpace_>(lhs.function_space(), lhs.coeff() + rhs.coeff());
    }
    friend constexpr FeFunction<FeSpace_> operator-(FeFunction<FeSpace_>& lhs, FeFunction<FeSpace_>& rhs) {
        return FeFunction<FeSpace_>(lhs.function_space(), lhs.coeff() - rhs.coeff());
    }
    // assignment from expansion coefficient vector
    FeFunction& operator=(const Eigen::Matrix<double, Dynamic, 1>& coeff) {
        fdapde_assert(coeff.size() > 0 && coeff.size() == fe_space_->n_dofs());
        coeff_ = coeff;
        return *this;
    }
   private:
    Eigen::Matrix<double, Dynamic, 1> coeff_;
    pointer_t fe_space_;
};

// given a not fe_assembler_packet callable type Derived_, builds a map from a discrete set of points (e.g., quadrature
// nodes) to the evaluation of Derived_ at that points, so that the results is fe_assembler_packet evaluable
template <typename Derived_>
struct FeMap :
    public std::conditional_t<
      internals::is_scalar_field_v<Derived_>, ScalarFieldBase<Derived_::StaticInputSize, FeMap<Derived_>>,
      MatrixFieldBase<Derived_::StaticInputSize, FeMap<Derived_>>> {
   private:
    static constexpr bool is_scalar = internals::is_scalar_field_v<Derived_>;
    using Derived = std::decay_t<Derived_>;
   public:
    using InputType = internals::fe_assembler_packet<Derived::StaticInputSize>;
    using Scalar = double;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    using Base = std::conditional_t<
      is_scalar, ScalarFieldBase<StaticInputSize, FeMap<Derived>>, MatrixFieldBase<StaticInputSize, FeMap<Derived>>>;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits | int(fe_assembler_flags::compute_physical_quad_nodes);
    static constexpr int ReadOnly = 1;
    static constexpr int Rows = []() { if constexpr(is_scalar) return 1; else return Derived::Rows; }();
    static constexpr int Cols = []() { if constexpr(is_scalar) return 1; else return Derived::Cols; }();

    constexpr FeMap() = default;
    constexpr FeMap(const Derived_& xpr) : xpr_(xpr) { }
    template <typename CellIterator>
    void init(
      const Eigen::Matrix<double, Dynamic, Dynamic>& nodes, [[maybe_unused]] CellIterator begin,
      [[maybe_unused]] CellIterator end) const {
        map_.resize(nodes.rows(), Rows * Cols);
        if constexpr (is_scalar) {
            for (int i = 0, n = nodes.rows(); i < n; ++i) { map_(i, 0) = xpr_(nodes.row(i)); }
        } else {
            for (int i = 0, n = nodes.rows(); i < n; ++i) {
                auto tmp = xpr_(nodes.row(i));
                if constexpr (Cols == 1) {
                    for (int j = 0; j < tmp.size(); ++j) { map_(i, j) = tmp[j]; }
                } else {   // tmp is a matrix
                    for (int j = 0; j < tmp.rows(); ++j) {
                        for (int k = 0; k < tmp.cols(); ++k) { map_(i, j) = tmp(j, k); }
                    }
                }
            }
        }
	return;
    }
    // fe assembler evaluation
    constexpr auto operator()(const InputType& fe_packet) const {
        if constexpr (is_scalar) {
            return map_(fe_packet.quad_node_id, 0);
        } else {
            if constexpr (Cols == 1) {
                return map_.row(fe_packet.quad_node_id);
            } else {   // reshape the flattened matrix to its correct Rows x Cols format
                return Eigen::Matrix<double, Rows, Cols, Eigen::RowMajor>(map_.row(fe_packet.quad_node_id));
            }
        }
    }
    constexpr auto eval(int i, const InputType& fe_packet) const {
        fdapde_static_assert(Rows != 1 && Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTOR_FIELDS);
        return map_(fe_packet.quad_node_id, i);
    }
    constexpr auto eval(int i, int j, const InputType& fe_packet) const {
        fdapde_static_assert(Rows != 1 && Cols != 1, THIS_METHOD_IS_ONLY_FOR_MATRIX_FIELDS);
        return map_(fe_packet.quad_node_id, i * Rows + j);
    }
    constexpr const Derived& derived() const { return xpr_; }
    constexpr int input_size() const { return StaticInputSize; }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
   private:
    Derived xpr_;
    mutable Eigen::Matrix<Scalar, Dynamic, Dynamic> map_;
};
  
// FeMap specialization for FeFunction types
template <typename FeSpace>
class FeMap<FeFunction<FeSpace>> : public ScalarFieldBase<FeSpace::embed_dim, FeMap<FeFunction<FeSpace>>> {
    using Derived = FeFunction<FeSpace>;
   public:
    using Base = ScalarFieldBase<FeSpace::embed_dim, FeMap<Derived>>;
    using InputType = internals::fe_assembler_packet<Derived::StaticInputSize>;
    using Scalar = double;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits | int(fe_assembler_flags::compute_physical_quad_nodes);

    FeMap() = default;
    FeMap(const Derived& xpr) : xpr_(std::addressof(xpr)) { }
    // fast evaluation routine which bypasses the point location step (quadrature nodes are not arbitrary points)
    template <typename CellIterator>
    void init(const Eigen::Matrix<double, Dynamic, Dynamic>& nodes, CellIterator begin, CellIterator end) const {
        map_.resize(nodes.rows(), 1);
        int n_cells = end.index() - begin.index();
        int n_quad_nodes = nodes.rows() / n_cells;
        int cell_id = begin.index();
        for (int i = 0, n = nodes.rows(); i < n; ++i) {
            // map node to reference cell and evaluate
            auto cell = xpr_->function_space().dof_handler().cell(cell_id + i / n_quad_nodes);
            Eigen::Matrix<double, FeSpace::local_dim, 1> ref_node =
              cell.invJ() * (nodes.row(i).transpose() - cell.node(0));
            Eigen::Matrix<int, Dynamic, 1> active_dofs = cell.dofs();
            Scalar value = 0;
            for (int i = 0, n = xpr_->function_space().n_shape_functions(); i < n; ++i) {
                value += xpr_->coeff()[active_dofs[i]] * xpr_->function_space().eval_shape_value(i, ref_node);
            }
            map_(i, 0) = value;
        }
    }
    // fe assembler evaluation
    constexpr Scalar operator()(const InputType& fe_packet) const { return map_(fe_packet.quad_node_id, 0); }
    constexpr const Derived& derived() const { return xpr_; }
    constexpr int input_size() const { return StaticInputSize; }
   private:
    const Derived* xpr_;
    mutable Eigen::Matrix<Scalar, Dynamic, Dynamic> map_;
};
  
// wraps an already evaluated field at (quadrature) nodes into a fe_assembler_packet callable object
template <int StaticInputSize_, int Rows_, int Cols_, typename DataT>
    requires(
      StaticInputSize_ > 0 && Rows_ > 0 && Cols_ > 0 &&
      ((Rows_ == 1 && Cols_ == 1 && internals::is_vector_like_v<DataT>) || internals::is_matrix_like_v<DataT>))
class FeCoeff :
    public std::conditional_t<
      Rows_ == 1 && Cols_ == 1, ScalarFieldBase<StaticInputSize_, FeCoeff<StaticInputSize_, Rows_, Cols_, DataT>>,
      MatrixFieldBase<StaticInputSize_, FeCoeff<StaticInputSize_, Rows_, Cols_, DataT>>> {
   private:
    using Derived = FeMap<DataT>;
    static constexpr bool is_scalar = (Rows_ == 1 && Cols_ == 1);
   public:
    using InputType = internals::fe_assembler_packet<StaticInputSize_>;
    using Scalar = double;
    static constexpr int StaticInputSize = StaticInputSize_;
    using Base = std::conditional_t<
      Rows_ == 1 && Cols_ == 1, ScalarFieldBase<StaticInputSize_, FeCoeff<StaticInputSize_, Rows_, Cols_, DataT>>,
      MatrixFieldBase<StaticInputSize_, FeCoeff<StaticInputSize_, Rows_, Cols_, DataT>>>;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = int(fe_assembler_flags::compute_physical_quad_nodes);
    static constexpr int ReadOnly = 1;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;

    constexpr FeCoeff() = default;
    template <typename DataT_>
        requires(std::is_convertible_v<DataT_, DataT>)
    constexpr FeCoeff(const DataT_& data) : data_(data) { }
    // fe assembler evaluation
    constexpr auto operator()(const InputType& fe_packet) const {
        if constexpr (is_scalar) {
            return data_[fe_packet.quad_node_id];
        } else {
            Eigen::Matrix<Scalar, Rows, Cols> tmp;
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < Cols; ++j) { tmp(i, j) = data_(fe_packet.quad_node_id, i * Cols + j); }
            }
            return tmp;
        }
    }
    constexpr auto eval(int i, const InputType& fe_packet) const {
        fdapde_static_assert(Rows != 1 && Cols == 1, THIS_METHOD_IS_FOR_VECTOR_FIELDS_ONLY);
        return data_(fe_packet.quad_node_id, i);
    }
    constexpr auto eval(int i, int j, const InputType& fe_packet) const {
        return data_(fe_packet.quad_node_id, i * Cols + j);
    }
    constexpr int input_size() const { return StaticInputSize; }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    const DataT& data() const { return data_; }
   private:
    DataT data_;
};

}   // namespace fdapde

#endif   // __FDAPDE_FE_OBJECTS_H__
