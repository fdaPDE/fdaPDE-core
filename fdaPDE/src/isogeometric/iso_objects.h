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

#ifndef __FDAPDE_ISO_OBJECTS_H__
#define __FDAPDE_ISO_OBJECTS_H__

#include "header_check.h"

namespace fdapde {  
namespace internals {

template <typename IsoSpace_>
struct iso_scalar_test_function_impl : public ScalarFieldBase<IsoSpace_::embed_dim, TestFunction<IsoSpace_, iso_tag>> {
    using TestSpace = std::decay_t<IsoSpace_>;
    using Base = ScalarFieldBase<IsoSpace_::embed_dim, TestFunction<IsoSpace_, iso_tag>>;
    using InputType = internals::iso_assembler_packet<TestSpace::embed_dim>;
    using Scalar = double; 
    static constexpr int StaticInputSize = TestSpace::embed_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | int(iso_assembler_flags::compute_shape_values);

    private:
     template<typename Derived_>
     struct FirstPartialDerivative_ : ScalarFieldBase<TestSpace::embed_dim, FirstPartialDerivative_<Derived_>> {
        using Derived = Derived_;
        template <typename T> using Meta = FirstPartialDerivative_<T>;
        using TestSpace = std::decay_t<IsoSpace_>;   // required from xpr_query<>
        using Base = ScalarFieldBase<TestSpace::embed_dim, FirstPartialDerivative_<Derived_>>;
        using InputType = internals::iso_assembler_packet<TestSpace::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = TestSpace::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(iso_assembler_flags::compute_shape_grad);

        FirstPartialDerivative_() noexcept = default;
        FirstPartialDerivative_(const Derived_& xpr) noexcept : xpr_(xpr), i_(0) { }
        FirstPartialDerivative_(const Derived_& xpr, int i) noexcept : xpr_(xpr), i_(i) { }
        // assembly evaluation
        constexpr Scalar operator()(const InputType& iso_packet) const { return iso_packet.test_grad(i_); }
        constexpr TestSpace& function_space() { return *(xpr_.iso_space_); }
        constexpr const TestSpace& function_space() const { return *(xpr_.iso_space_); }
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
       private:
        int i_;
        Derived xpr_;
     };

     template <typename Derived_>
    struct MixedPartialDerivative_ : ScalarFieldBase<TestSpace::embed_dim, MixedPartialDerivative_<Derived_>> {
        using Derived = Derived_;
        template <typename T> using Meta = MixedPartialDerivative_<T>;      
        using TestSpace = std::decay_t<IsoSpace_>;   // required from xpr_query<>
        using Base = ScalarFieldBase<TestSpace::embed_dim, MixedPartialDerivative_<Derived_>>;
        using InputType = internals::iso_assembler_packet<TestSpace::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = TestSpace::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(iso_assembler_flags::compute_shape_hess);

        MixedPartialDerivative_() noexcept = default;
        MixedPartialDerivative_(const Derived_& xpr) noexcept : xpr_(xpr), i_(0), j_(0) { }
        MixedPartialDerivative_(const Derived_& xpr, int i, int j) noexcept : xpr_(xpr), i_(i), j_(j) { }

        // assembly evaluation
        constexpr Scalar operator()(const InputType& iso_packet) const { return iso_packet.test_hess(i_,j_); }
        constexpr TestSpace& function_space() { return *(xpr_.iso_space_); }
        constexpr const TestSpace& function_space() const { return *(xpr_.iso_space_); }
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
       private:
        int i_, j_;
        Derived xpr_;
    };

    public:

    // expose derivative types
    using FirstPartialDerivative  = FirstPartialDerivative_ <TestFunction<IsoSpace_, iso_tag>>;
    using MixedPartialDerivative = MixedPartialDerivative_<TestFunction<IsoSpace_, iso_tag>>;

    constexpr iso_scalar_test_function_impl() noexcept = default;
    constexpr iso_scalar_test_function_impl(IsoSpace_& iso_space) noexcept : iso_space_(std::addressof(iso_space)) { }  
    // assembly evaluation
    constexpr Scalar operator()(const InputType& iso_packet) const { return iso_packet.test_value; }
    constexpr TestSpace& function_space() { return *iso_space_; }
    constexpr const TestSpace& function_space() const { return *iso_space_; }
    constexpr int input_size() const { return StaticInputSize; }
   private:
    TestSpace* iso_space_;


};

template <typename IsoSpace_>
struct iso_scalar_trial_function_impl : public ScalarFieldBase<IsoSpace_::embed_dim, TrialFunction<IsoSpace_, iso_tag>> {
    using TrialSpace = std::decay_t<IsoSpace_>;
    using Base = ScalarFieldBase<IsoSpace_::embed_dim, TrialFunction<IsoSpace_, iso_tag>>;
    using InputType = internals::iso_assembler_packet<TrialSpace::embed_dim>;
    using Scalar = double;
    static constexpr int StaticInputSize = TrialSpace::embed_dim;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0 | int(iso_assembler_flags::compute_shape_values);
    private:
    //definitions of derivative functors
    template<typename Derived_>
    struct FirstPartialDerivative_ : ScalarFieldBase<TrialSpace::embed_dim, FirstPartialDerivative_<Derived_>> {
        using Derived = Derived_;
        template <typename T> using Meta = FirstPartialDerivative_<T>;
        using TrialSpace = std::decay_t<IsoSpace_>;   // required from xpr_query<>
        using Base = ScalarFieldBase<TrialSpace::embed_dim, FirstPartialDerivative_<Derived_>>;
        using InputType = internals::iso_assembler_packet<TrialSpace::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = TrialSpace::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(iso_assembler_flags::compute_shape_grad);

        FirstPartialDerivative_() noexcept = default;
        FirstPartialDerivative_(const Derived_& xpr) noexcept : xpr_(xpr), i_(0) { }
        FirstPartialDerivative_(const Derived_& xpr, int i) noexcept : xpr_(xpr), i_(i) { }
        // assembly evaluation
        constexpr Scalar operator()(const InputType& iso_packet) const { return iso_packet.trial_grad(i_); }
        constexpr TrialSpace& function_space() { return *(xpr_.iso_space_); }
        constexpr const TrialSpace& function_space() const { return *(xpr_.iso_space_); }
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
       private:
        int i_;
        Derived xpr_;
    };

    template <typename Derived_>
    struct MixedPartialDerivative_ : ScalarFieldBase<TrialSpace::embed_dim, MixedPartialDerivative_<Derived_>> {
        using Derived = Derived_;
        template <typename T> using Meta = MixedPartialDerivative_<T>;      
        using TestSpace = std::decay_t<IsoSpace_>;   // required from xpr_query<>
        using Base = ScalarFieldBase<TrialSpace::embed_dim, MixedPartialDerivative_<Derived_>>;
        using InputType = internals::iso_assembler_packet<TrialSpace::embed_dim>;
        using Scalar = double;
        static constexpr int StaticInputSize = TrialSpace::embed_dim;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0 | int(iso_assembler_flags::compute_shape_hess);

        MixedPartialDerivative_() noexcept = default;
        MixedPartialDerivative_(const Derived_& xpr) noexcept : xpr_(xpr), i_(0), j_(0) { }
        MixedPartialDerivative_(const Derived_& xpr, int i, int j) noexcept : xpr_(xpr), i_(i), j_(j) {
         }

        // assembly evaluation
        constexpr Scalar operator()(const InputType& iso_packet) const { return iso_packet.trial_hess(i_,j_); }
        constexpr TrialSpace& function_space() { return *(xpr_.iso_space_); }
        constexpr const TrialSpace& function_space() const { return *(xpr_.iso_space_); }
        constexpr int input_size() const { return StaticInputSize; }
        constexpr const Derived& derived() const { return xpr_; }
       private:
        int i_, j_;
        Derived xpr_;
    };

    public:

    // expose derivative types
    using FirstPartialDerivative  = FirstPartialDerivative_ <TrialFunction<IsoSpace_, iso_tag>>;
    using MixedPartialDerivative = MixedPartialDerivative_<TrialFunction<IsoSpace_, iso_tag>>;

    constexpr iso_scalar_trial_function_impl() noexcept = default;
    constexpr iso_scalar_trial_function_impl(IsoSpace_& iso_space) noexcept : iso_space_(std::addressof(iso_space)) { }  
    // assembly evaluation
    constexpr Scalar operator()(const InputType& iso_packet) const { return iso_packet.trial_value; }
    constexpr TrialSpace& function_space() { return *iso_space_; }
    constexpr const TrialSpace& function_space() const { return *iso_space_; }
    constexpr int input_size() const { return StaticInputSize; }
    
    private:
    TrialSpace* iso_space_;
    
};

} // namespace internals

template<typename IsoSpace_>
    requires(std::is_same_v<typename std::decay_t<IsoSpace_>::discretization_category, iso_tag>)
struct TestFunction<IsoSpace_, iso_tag> : public internals::iso_scalar_test_function_impl<IsoSpace_> {
    using Base = internals::iso_scalar_test_function_impl<IsoSpace_>;
    constexpr TestFunction() = default;
    constexpr TestFunction(IsoSpace_& iso_space) : Base(iso_space) { }
};

// grad, div ????


// partial derivatives of scalar test function
template <typename IsoSpace_>
struct PartialDerivative<TestFunction<IsoSpace_, iso_tag>, 1> :
    public TestFunction<IsoSpace_, iso_tag>::FirstPartialDerivative {
    PartialDerivative() = default;
    PartialDerivative(const TestFunction<IsoSpace_, iso_tag>& f, int i) :
        TestFunction<IsoSpace_, iso_tag>::FirstPartialDerivative(f,i) { }
};
template <typename IsoSpace_>
struct PartialDerivative<TestFunction<IsoSpace_, iso_tag>, 2> :
    public TestFunction<IsoSpace_, iso_tag>::MixedPartialDerivative {
    PartialDerivative() = default;
    PartialDerivative(const TestFunction<IsoSpace_, iso_tag>& f, int i, int j) :
        TestFunction<IsoSpace_, iso_tag>::MixedPartialDerivative(f,i,j) { }
};

// trial function struct
template <typename IsoSpace_>
    requires(std::is_same_v<typename std::decay_t<IsoSpace_>::discretization_category, iso_tag>)
struct TrialFunction<IsoSpace_, iso_tag> : public internals::iso_scalar_trial_function_impl<IsoSpace_> {
    using Base = internals::iso_scalar_trial_function_impl<IsoSpace_>;
    using TrialSpace = typename Base::TrialSpace;
    static constexpr int local_dim = IsoSpace_::local_dim;
    static constexpr int embed_dim = IsoSpace_::embed_dim;
    
    constexpr TrialFunction() = default;
    constexpr TrialFunction(IsoSpace_& iso_space) : Base(iso_space) { }
    // norm evaluation
    double l2_squared_norm() {
        TrialFunction u(*Base::iso_space_);
        TestFunction  v(*Base::iso_space_);
        auto assembler = integrate(*Base::iso_space_->mesh())(u * v);
        return coeff_.dot(assembler.assemble() * coeff_);
    }
    double l2_norm() {return std::sqrt(l2_squared_norm());} 
    const Eigen::Matrix<double, Dynamic, 1>& coeff() const { return coeff_; }
    void set_coeff(const Eigen::Matrix<double, Dynamic, 1>& coeff) { coeff_ = coeff; }
   private:
    Eigen::Matrix<double, Dynamic, 1> coeff_; // ??????
};

// partial derivatives of scalar trial function
template <typename IsoSpace_>
struct PartialDerivative<TrialFunction<IsoSpace_, iso_tag>, 1> :
    public TrialFunction<IsoSpace_, iso_tag>::FirstPartialDerivative {
    PartialDerivative() = default;
    PartialDerivative(const TrialFunction<IsoSpace_, iso_tag>& f, int i) :
        TrialFunction<IsoSpace_, iso_tag>::FirstPartialDerivative(f,i) { }
};
template <typename IsoSpace_>
struct PartialDerivative<TrialFunction<IsoSpace_, iso_tag>, 2> :
    public TrialFunction<IsoSpace_, iso_tag>::MixedPartialDerivative {
    PartialDerivative() = default;
    PartialDerivative(const TrialFunction<IsoSpace_, iso_tag>& f, int i, int j) :
        TrialFunction<IsoSpace_, iso_tag>::MixedPartialDerivative(f, i, j) {
         }
};

// alisas ???? dx ddx

// representation of u(x) = \sum_{i=1}^{n_dofs} u_i \psi_i(x) with \{ \psi_i \}_i a NURBS basis system
template <typename IsoSpace_> class IsoFunction : public ScalarFieldBase<IsoSpace_::local_dim, IsoFunction<IsoSpace_>> {
    using IsoMesh = typename IsoSpace_::IsoMesh;
    public:
    using IsoSpace = std::decay_t<IsoSpace_>;
    using Base = ScalarFieldBase<IsoSpace_::local_dim, IsoFunction<IsoSpace_>>;
    using DofHandlerType = typename IsoSpace::DofHandlerType;
    using InputType = Eigen::Matrix<double, IsoSpace::local_dim, 1>;
    using Scalar = double;
    static constexpr int StaticInputSize = IsoSpace::local_dim;
    static constexpr int Rows = 1;
    static constexpr int Cols = 1;
    static constexpr int NestAsRef = 1;
    static constexpr int local_dim = IsoMesh::local_dim;
    static constexpr int embed_dim = IsoMesh::embed_dim;
    static constexpr int XprBits = 0;

    IsoFunction() = default;
    explicit IsoFunction(IsoSpace_& iso_space) : iso_space_(&iso_space) {
        coeff_ = Eigen::Matrix<double, Dynamic, 1>::Zero(iso_space_->n_dofs());
    } 
    IsoFunction(IsoSpace_& sp_space, const Eigen::Matrix<double, Dynamic, 1>& coeff) :
        iso_space_(std::addressof(sp_space)), coeff_(coeff) {
        fdapde_assert(coeff.size() > 0 && coeff.size() == iso_space_->n_dofs());
    }

    Scalar operator()(const InputType& p){
        int e_id = iso_space_->mesh().locate_param(p);
        if (e_id == -1) return std::numeric_limits<Scalar>::quiet_NaN();   // return NaN if point lies outside domain
        // map p to reference cell and evaluate
        typename DofHandlerType::CellType cell = iso_space_->dof_handler().cell(e_id);
        InputType ref_p = cell.inverse_affine_map(p) ; // da capire
        std::vector<int> active_dofs = cell.dofs();

        Scalar value = 0;
        for (int i = 0, n = active_dofs.size(); i < n; ++i) {
            value += coeff_[active_dofs[i]] * iso_space_->eval_shape_value(active_dofs[i], p); // perchy ref p ???
        }
        return value;
    }

    Eigen::Matrix<double, embed_dim, 1> phys_grad(const InputType& p) const{
        int e_id = iso_space_->mesh().locate_param(p);
        if (e_id == -1) return Eigen::Matrix<double, embed_dim, 1>::Zero();
        // map p to reference cell and evaluate
        typename DofHandlerType::CellType cell = iso_space_->dof_handler().cell(e_id);
        Eigen::Matrix<double, embed_dim, local_dim> F = iso_space_->mesh().eval_param_derivatives(p).first_derivative;
        InputType ref_p = cell.inverse_affine_map(p);
        std::vector<int> active_dofs = cell.dofs();

        Eigen::Matrix<double, local_dim, 1> grad = Eigen::Matrix<double, local_dim, 1>::Zero();
        for (int i = 0, n = active_dofs.size(); i < n; ++i) {
            grad += coeff_[active_dofs[i]] * iso_space_->eval_shape_grad(active_dofs[i], p);
        }
        return F * (F.transpose() * F).inverse() * grad;
    }

    Eigen::Matrix<double, embed_dim, embed_dim> phys_hess(const InputType& p) const{
        int e_id = iso_space_->mesh().locate_param(p);
        if (e_id == -1) return Eigen::Matrix<double, embed_dim, embed_dim>::Zero();
        // map p to reference cell and evaluate
        typename DofHandlerType::CellType cell = iso_space_->dof_handler().cell(e_id);
        auto ders = iso_space_->mesh().eval_param_derivatives(p,true);

        Eigen::Matrix<double, embed_dim, local_dim> F = ders.first_derivative;
        MdArray<double, MdExtents<embed_dim, local_dim, local_dim>> param_hess = *(ders.second_derivative);

        std::vector<int> active_dofs = cell.dofs();

        Eigen::Matrix<double, local_dim, local_dim> hess = Eigen::Matrix<double, local_dim, local_dim>::Zero();
        for (int i = 0, n = active_dofs.size(); i < n; ++i) {
            hess += coeff_[active_dofs[i]] * iso_space_->eval_shape_hess(active_dofs[i], p);  
        }

        Eigen::Matrix<double, embed_dim, embed_dim> phys_hess_;

        Eigen::Matrix<double, local_dim, embed_dim> dxi_dx = (F.transpose() * F).inverse() * F.transpose();
        phys_hess_ = dxi_dx.transpose() * hess * dxi_dx;


        Eigen::Matrix<double, local_dim, 1> grad = Eigen::Matrix<double, local_dim, 1>::Zero();
        for (int i = 0, n = active_dofs.size(); i < n; ++i) {
            grad += coeff_[active_dofs[i]] * iso_space_->eval_shape_grad(active_dofs[i], p);
        }

        for (int ii = 0; ii < embed_dim; ++ii) {
            for (int jj = 0; jj < embed_dim; ++jj) {
                for (int alpha = 0; alpha < local_dim; ++alpha) {
                    double d2xi = 0.0;
                    for (int kk = 0; kk < embed_dim; ++kk) {
                        for (int beta = 0; beta < local_dim; ++beta) {
                            for (int gamma = 0; gamma < local_dim; ++gamma) {
                                d2xi -= dxi_dx(alpha, kk) * param_hess(kk, beta, gamma) * dxi_dx(beta, ii) * dxi_dx(gamma, jj); 
                            }
                        }
                    }
                    phys_hess_(ii, jj) += grad(alpha) * d2xi;
                }
            }
        }        
        Eigen::Matrix<double, embed_dim, embed_dim> P;
        if constexpr(embed_dim == 2){
        P = Eigen::Matrix<double, embed_dim, embed_dim>::Identity();
        }
        else{
        Eigen::Matrix<double, embed_dim, 1> n = ((F.col(0)).cross(F.col(1))).normalized();
        P = Eigen::Matrix<double, embed_dim, embed_dim>::Identity() - n * n.transpose();
        }
        return P * phys_hess_ * P;
    }





    // norm evaluation
    double l2_squared_norm() {
        TrialFunction u(*iso_space_);
        TestFunction  v(*iso_space_);
        auto assembler = integrate(*(iso_space_->mesh()))(u * v);
        return coeff_.dot(assembler.assemble() * coeff_);
    }
    double l2_norm() { return std::sqrt(l2_squared_norm()); }
    // getters
    const Eigen::Matrix<double, Dynamic, 1>& coeff() const { return coeff_; }
    constexpr IsoSpace& function_space() { return *iso_space_; }
    constexpr const IsoSpace& function_space() const { return *iso_space_; }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr int input_size() const { return StaticInputSize; }
    void set_coeff(const Eigen::Matrix<double, Dynamic, 1>& coeff) { coeff_ = coeff; }
    // linear algebra between iso functions
    friend constexpr IsoFunction<IsoSpace_> operator+(IsoFunction<IsoSpace_>& lhs, IsoFunction<IsoSpace_>& rhs) {
        return IsoFunction<IsoSpace_>(lhs.function_space(), lhs.coeff() + rhs.coeff());
    }
    friend constexpr IsoFunction<IsoSpace_> operator-(IsoFunction<IsoSpace_>& lhs, IsoFunction<IsoSpace_>& rhs) {
        return IsoFunction<IsoSpace_>(lhs.function_space(), lhs.coeff() - rhs.coeff());
    }
    // assignment from expansion coefficient vector
    IsoFunction& operator=(const Eigen::Matrix<double, Dynamic, 1>& coeff) {
        fdapde_assert(coeff.size() > 0 && coeff.size() == iso_space_->n_dofs());
        coeff_ = coeff;
        return *this;
    }

    private:
    Eigen::Matrix<double, Dynamic, 1> coeff_;
    IsoSpace* iso_space_;
};


// given a not iso_assembler_packet callable type Derived_, builds a map from a discrete set of points (e.g., quadrature
// nodes) to the evaluation of Derived_ at that points, so that the results is sp_assembler_packet evaluable
// IsoMap specialization for FeFunction types

template <typename Derived_>
struct IsoMap :
    public std::conditional_t<
      internals::is_scalar_field_v<Derived_>, ScalarFieldBase<Derived_::StaticInputSize, IsoMap<Derived_>>,
      MatrixFieldBase<Derived_::StaticInputSize, IsoMap<Derived_>>> {
   private:
    static constexpr bool is_scalar = internals::is_scalar_field_v<Derived_>;
    using Derived = std::decay_t<Derived_>;
   public:
    using InputType = internals::iso_assembler_packet<Derived::StaticInputSize>;
    using Scalar = double;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    using Base = std::conditional_t<
      is_scalar, ScalarFieldBase<StaticInputSize, IsoMap<Derived>>, MatrixFieldBase<StaticInputSize, IsoMap<Derived>>>;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits | int(iso_assembler_flags::compute_physical_quad_nodes);
    static constexpr int ReadOnly = 1;
    static constexpr int Rows = []() { if constexpr(is_scalar) return 1; else return Derived::Rows; }();
    static constexpr int Cols = []() { if constexpr(is_scalar) return 1; else return Derived::Cols; }();

    constexpr IsoMap() = default;
    constexpr IsoMap(const Derived_& xpr) : xpr_(xpr) { }
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
    constexpr auto operator()(const InputType& iso_packet) const {
        if constexpr (is_scalar) {
            return map_(iso_packet.quad_node_id, 0);
        } else {
            if constexpr (Cols == 1) {
                return map_.row(iso_packet.quad_node_id);
            } else {   // reshape the flattened matrix to its correct Rows x Cols format
                return Eigen::Matrix<double, Rows, Cols, Eigen::RowMajor>(map_.row(iso_packet.quad_node_id));
            }
        }
    }
    constexpr auto eval(int i, const InputType& iso_packet) const {
        fdapde_static_assert(Rows != 1 && Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTOR_FIELDS);
        return map_(iso_packet.quad_node_id, i);
    }
    constexpr auto eval(int i, int j, const InputType& iso_packet) const {
        fdapde_static_assert(Rows != 1 && Cols != 1, THIS_METHOD_IS_ONLY_FOR_MATRIX_FIELDS);
        return map_(iso_packet.quad_node_id, i * Rows + j);
    }
    constexpr const Derived& derived() const { return xpr_; }
    constexpr int input_size() const { return StaticInputSize; }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
   private:
    Derived xpr_;
    mutable Eigen::Matrix<Scalar, Dynamic, Dynamic> map_;
};
#endif








} // namespace fdapde