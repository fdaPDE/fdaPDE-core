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

#ifndef __FE_VECTOR_ASSEMBLER_H__
#define __FE_VECTOR_ASSEMBLER_H__

#include <unordered_map>

#include "fe_assembler_base.h"

namespace fdapde {
namespace internals {

// base class for vector finite element assembly loops
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
struct vector_fe_assembler_base {
    // detect test space (since a test function is always present in a weak form)
    using TestSpace = std::decay_t<decltype(test_space(std::declval<Form_>()))>;
    fdapde_static_assert(
      TestSpace::n_components != 1, YOU_PROVIDED_A_SCALAR_VALUED_FUNCTION_SPACE_TO_A_VECTOR_ASSEMBLY_LOOP);
    using Form = std::decay_t<decltype(meta::xpr_wrap<FeMap, decltype([]<typename Xpr>() {
          return !std::is_invocable_v<
            Xpr, fe_assembler_packet<Xpr::StaticInputSize>>;
        })>(std::declval<Form_>()))>;
    using Triangulation = typename std::decay_t<Triangulation_>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int Options = Options_;
    using FunctionSpace = TestSpace;
    using FeType = typename FunctionSpace::FeType;
    using fe_traits = std::conditional_t<
      Options == CellMajor, fe_cell_assembler_traits<FunctionSpace, Quadrature_...>,
      fe_face_assembler_traits<FunctionSpace, Quadrature_...>>;
    using Quadrature = typename fe_traits::Quadrature;
    using dof_descriptor = typename fe_traits::dof_descriptor;
    using BasisType = typename dof_descriptor::BasisType;
    static constexpr int n_quadrature_nodes = Quadrature::n_nodes;
    static constexpr int n_basis = BasisType::n_basis;
    static constexpr int n_components = FunctionSpace::n_components;
  
    vector_fe_assembler_base() = default;
    vector_fe_assembler_base(
      const Form_& form, typename fe_traits::geo_iterator begin, typename fe_traits::geo_iterator end,
      const Quadrature_&... quadrature)
        requires(sizeof...(quadrature) <= 1) :
        form_(meta::xpr_wrap<FeMap, decltype([]<typename Xpr>() {
	      return !std::is_invocable_v<Xpr, fe_assembler_packet<Xpr::StaticInputSize>>;
	    })>(form)),
        quadrature_([... quadrature = std::forward<const Quadrature_>(quadrature)]() {
            if constexpr (sizeof...(quadrature) == 1) {
                return std::get<0>(std::make_tuple(quadrature...));
            } else {
                return Quadrature {};
            }
        }()),
        dof_handler_(&test_space(form_).dof_handler()),
        begin_(begin),
        end_(end) {
        fdapde_assert(dof_handler_->n_dofs() > 0);
    }
   protected:
    // compile-time evaluation of \psi_i(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    template <typename Quadrature__, typename dof_descriptor__>
    static consteval MdArray<
      double, MdExtents<dof_descriptor__::BasisType::n_basis, Quadrature__::n_nodes, n_components>>
    eval_shape_values() {
        using BasisType = typename dof_descriptor__::BasisType;
        constexpr int n_basis = BasisType::n_basis;
        constexpr int n_quadrature_nodes = Quadrature__::n_nodes;
        MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components>> shape_values_ {};
        BasisType basis {dof_descriptor__().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            for (int j = 0; j < n_quadrature_nodes; ++j) {
                auto value = basis[i](Quadrature::nodes.row(j).transpose());
                for (int k = 0; k < n_components; ++k) { shape_values_(i, j, k) = value[k]; }
            }
        }
        return shape_values_;
    }
    // compile-time evaluation of \nabla{\psi_i}(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    template <typename Quadrature__, typename dof_descriptor__>
    static consteval MdArray<
      double, MdExtents<dof_descriptor__::BasisType::n_basis, Quadrature__::n_nodes, n_components, local_dim>>
    eval_shape_grads() {
        using BasisType = typename dof_descriptor__::BasisType;
        constexpr int n_basis = BasisType::n_basis;
        constexpr int n_quadrature_nodes = Quadrature__::n_nodes;
        MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components, local_dim>> shape_grad_ {};
        BasisType basis {dof_descriptor__().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            for (int j = 0; j < n_quadrature_nodes; ++j) {
                auto grad = basis[i].gradient()(Quadrature::nodes.row(j).transpose());
                for (int k = 0; k < n_components; ++k) {
                    for (int h = 0; h < local_dim; ++h) { shape_grad_(i, j, k, h) = grad(h, k); }
                }
            }
        }
        return shape_grad_;
    }

    // test basis functions evaluations
    static constexpr MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components>> test_shape_values_ =
      eval_shape_values<Quadrature, dof_descriptor>();
    static constexpr MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components, local_dim>>
      test_shape_grads_ = eval_shape_grads<Quadrature, dof_descriptor>();

    void distribute_quadrature_nodes(
      std::unordered_map<const void*, DMatrix<double>>& fe_map_buff, typename fe_traits::dof_iterator begin,
      typename fe_traits::dof_iterator end) const {
        DMatrix<double> quad_nodes;
        quad_nodes.resize(n_quadrature_nodes * (end_.index() - begin_.index()), embed_dim);
        int local_cell_id = 0;
        for (typename fe_traits::geo_iterator it = begin_; it != end_; ++it) {
            for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                quad_nodes.row(local_cell_id * n_quadrature_nodes + q_k) =
                  it->J() *
                    Eigen::Map<const SMatrix<n_quadrature_nodes, Quadrature::local_dim>>(quadrature_.nodes.data())
                      .row(q_k)
                      .transpose() +
                  it->node(0);
            }
            local_cell_id++;
        }
        // evaluate FeMap nodes at quadrature nodes
        meta::xpr_apply_if<
          decltype([]<typename Xpr_, typename... Args>(Xpr_& xpr, Args&&... args) {
              xpr.init(std::forward<Args>(args)...);
              return;
          }),
          decltype([]<typename Xpr_>() {
              return requires(Xpr_ xpr) { xpr.init(fe_map_buff, quad_nodes, begin, end); };
          })>(form_, fe_map_buff, quad_nodes, begin, end);
	return;
    }
    // moves \nabla{\psi_i}(q_k) from the reference cell to physical cell pointed by it
    template <typename CellIterator, typename RefGrads, typename Dst>
    constexpr void eval_shape_grads_on_cell(CellIterator& it, const RefGrads& ref_grads, Dst& dst) const {
        const int n_basis_ = ref_grads.extent(0);
        const int n_quadrature_nodes_ = ref_grads.extent(1);
        const int n_components_ = ref_grads.extent(2);

	for (int i = 0; i < n_basis_; ++i) {
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                // get i-th reference basis gradient evaluted at j-th quadrature node
                auto ref_grad = ref_grads.template slice<0, 1>(i, j).matrix();
                for (int k = 0; k < n_components_; ++k) {
                    dst.template slice<0, 1, 2>(i, j, k).assign_from(cexpr::Matrix<double, 1, local_dim>(
                      ref_grad.row(k) * cexpr::Map<const double, local_dim, embed_dim>(it->invJ().data())));
                }
            }
        }
        return;
    }
    // computes div(\psi_i)(q_k) on the physical cell pointer by it
    template <typename CellIterator, typename RefGrads, typename Dst>
    constexpr void eval_shape_div_on_cell(CellIterator& it, const RefGrads& ref_grads, Dst& dst) const {
        const int n_basis_ = ref_grads.extent(0);
        const int n_quadrature_nodes_ = ref_grads.extent(1);
        const int n_components_ = ref_grads.extent(2);
	fdapde_assert(n_components_ == local_dim);
	
        for (int i = 0; i < n_basis_; ++i) {
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                double div_ = 0;
                for (int k = 0; k < n_components_; ++k) {
                    // get k-th component of i-th reference basis gradient evaluated at j-th quadrature node
                    auto ref_grad = ref_grads.template slice<0, 1, 2>(i, j, k).matrix();
                    div_ += cexpr::Map<const double, local_dim, embed_dim>(it->invJ().data()).col(k).dot(ref_grad);
                }
                dst(i, j) = div_;
            }
        }
        return;
    }

    Form form_;
    Quadrature quadrature_ {};
    const DofHandler<local_dim, embed_dim>* dof_handler_;
    typename fe_traits::geo_iterator begin_, end_;
};
  
// galerkin and petrov-galerkin vector finite element assembly loop
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class vector_fe_bilinear_form_assembly_loop :
    public vector_fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>,
    public fe_assembly_xpr_base<vector_fe_bilinear_form_assembly_loop<Triangulation_, Form_, Options_, Quadrature_...>> {
    // detect trial and test spaces from bilinear form
    using TrialSpace = trial_space_t<Form_>;
    using TestSpace = test_space_t<Form_>;
    fdapde_static_assert(
      TrialSpace::local_dim == TestSpace::local_dim && TrialSpace::embed_dim == TestSpace::embed_dim,
      TRIAL_AND_TEST_SPACES_LOCAL_OR_EMBEDDING_DIMENSIONS_DO_NOT_MATCH);
    fdapde_static_assert(
      TestSpace::n_components != 1 || TrialSpace::n_components != 1,
      THIS_CLASS_IS_FOR_VECTOR_VALUED_FINITE_ELEMENTS_ONLY);
    static constexpr bool is_galerkin = std::is_same_v<TrialSpace, TestSpace>;
    static constexpr bool is_petrov_galerkin = !is_galerkin;
    using Base = vector_fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>;
    using Form = typename Base::Form;
    static constexpr int local_dim = Base::local_dim;
    static constexpr int embed_dim = Base::embed_dim;
    using Base::form_;

    // as trial and test spaces could be different, we here need to redefine some properties of Base
    // trial space properties
    using TrialFeType = typename TrialSpace::FeType;
    using trial_fe_traits = std::conditional_t<
      Options_ == CellMajor, fe_cell_assembler_traits<TrialSpace, Quadrature_...>,
      fe_face_assembler_traits<TrialSpace, Quadrature_...>>;
    using trial_dof_descriptor = typename trial_fe_traits::dof_descriptor;
    using TrialBasisType = typename trial_dof_descriptor::BasisType;
    static constexpr int n_trial_basis = TrialBasisType::n_basis;
    static constexpr int n_trial_components = TrialSpace::n_components;
    // test space properties
    using TestFeType = typename TestSpace::FeType;
    using test_fe_traits = std::conditional_t<
      Options_ == CellMajor, fe_cell_assembler_traits<TestSpace, Quadrature_...>,
      fe_face_assembler_traits<TestSpace, Quadrature_...>>;
    using test_dof_descriptor = typename test_fe_traits::dof_descriptor;
    using TestBasisType = typename test_dof_descriptor::BasisType;
    static constexpr int n_test_basis = TestBasisType::n_basis;
    static constexpr int n_test_components = TestSpace::n_components;

    using Quadrature = std::conditional_t<
      (is_galerkin || sizeof...(Quadrature_) > 0), typename Base::Quadrature,
      higher_order_fe_quadrature_t<
        typename TrialFeType::template select_cell_quadrature_t<local_dim>,
        typename TestFeType ::template select_cell_quadrature_t<local_dim>>>;
    static constexpr int n_quadrature_nodes = Quadrature::n_nodes;

    // selected Quadrature could be different than Base::Quadrature, evaluate trial and (re-evaluate) test functions
    static constexpr auto test_shape_values_  = Base::template eval_shape_values<Quadrature, test_dof_descriptor >();
    static constexpr auto trial_shape_values_ = Base::template eval_shape_values<Quadrature, trial_dof_descriptor>();
    static constexpr auto test_shape_grads_   = Base::template eval_shape_grads <Quadrature, test_dof_descriptor >();
    static constexpr auto trial_shape_grads_  = Base::template eval_shape_grads <Quadrature, trial_dof_descriptor>();
    // private data members
    const DofHandler<local_dim, embed_dim>* trial_dof_handler_;
    Quadrature quadrature_ {};
    constexpr const DofHandler<local_dim, embed_dim>* test_dof_handler() const { return Base::dof_handler_; }
    constexpr const DofHandler<local_dim, embed_dim>* trial_dof_handler() const {
        return is_galerkin ? Base::dof_handler_ : trial_dof_handler_;
    }
   public:
    vector_fe_bilinear_form_assembly_loop() = default;
    vector_fe_bilinear_form_assembly_loop(
      const Form_& form, typename Base::fe_traits::geo_iterator begin, typename Base::fe_traits::geo_iterator end,
      const Quadrature_&... quadrature) requires(sizeof...(quadrature) <= 1)
        : Base(form, begin, end, quadrature...) {
        if constexpr (is_petrov_galerkin) { trial_dof_handler_ = &trial_space(form_).dof_handler(); }
        fdapde_assert(test_dof_handler()->n_dofs() != 0 && trial_dof_handler()->n_dofs() != 0);
    }

    SpMatrix<double> assemble() const {
        SpMatrix<double> assembled_mat(test_dof_handler()->n_dofs(), trial_dof_handler()->n_dofs());
        std::vector<Eigen::Triplet<double>> triplet_list;
	assemble(triplet_list);
	// linearity of the integral is implicitly used here, as duplicated triplets are summed up (see Eigen docs)
        assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
        assembled_mat.makeCompressed();
        return assembled_mat;
    }
    void assemble(std::vector<Eigen::Triplet<double>>& triplet_list) const {
        using iterator = typename Base::fe_traits::dof_iterator;
        iterator begin(Base::begin_.index(), test_dof_handler(), Base::begin_.marker());
        iterator end  (Base::end_.index(),   test_dof_handler(), Base::end_.marker()  );
        // prepare assembly loop
        DVector<int> test_active_dofs, trial_active_dofs;
        MdArray<double, MdExtents<n_test_basis,  n_quadrature_nodes, n_test_components,  local_dim>> test_grads;
        MdArray<double, MdExtents<n_trial_basis, n_quadrature_nodes, n_trial_components, local_dim>> trial_grads;
	cexpr::Matrix<double, n_test_basis,  n_quadrature_nodes> test_divs;
	cexpr::Matrix<double, n_trial_basis, n_quadrature_nodes> trial_divs;
	
        std::unordered_map<const void*, DMatrix<double>> fe_map_buff;
        if constexpr (Form::XprBits & fe_assembler_flags::compute_physical_quad_nodes) {
            Base::distribute_quadrature_nodes(
              fe_map_buff, begin, end);   // distribute quadrature nodes on physical mesh (if required)
        }

        // start assembly loop
        internals::fe_assembler_packet<local_dim> fe_packet(n_trial_components, n_test_components);
	int local_cell_id = 0;
        for (iterator it = begin; it != end; ++it) {
            // update fe_packet content based on form requests
            fe_packet.cell_measure = it->measure();
            if constexpr (Form::XprBits & fe_assembler_flags::compute_cell_diameter) {
                fe_packet.cell_diameter = it->diameter();
            }
            if constexpr (Form::XprBits & fe_assembler_flags::compute_shape_grad) {
                Base::eval_shape_grads_on_cell(it, test_shape_grads_, test_grads);
                if constexpr (is_petrov_galerkin) Base::eval_shape_grads_on_cell(it, trial_shape_grads_, trial_grads);
            }
            if constexpr (Form::XprBits & fe_assembler_flags::compute_shape_div) {
                Base::eval_shape_div_on_cell(it, test_shape_grads_, test_divs);
                if constexpr (is_petrov_galerkin) Base::eval_shape_grads_on_cell(it, trial_shape_grads_, trial_divs);
            }

            // perform integration of weak form for (i, j)-th basis pair
            test_active_dofs = it->dofs();
            if constexpr (is_petrov_galerkin) { trial_active_dofs = trial_dof_handler()->active_dofs(it->id()); }
            for (int i = 0; i < n_trial_basis; ++i) {       // trial function loop
                for (int j = 0; j < n_test_basis; ++j) {   // test function loop
                    double value = 0;
                    for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                        if constexpr (Form::XprBits & fe_assembler_flags::compute_shape_values) {
                            fe_packet.trial_value = trial_shape_values_.template slice<0, 1>(i, q_k);
                            fe_packet.test_value  = test_shape_values_ .template slice<0, 1>(j, q_k);
                        }
                        if constexpr (Form::XprBits & fe_assembler_flags::compute_shape_grad) {
                            fe_packet.trial_grad = is_galerkin ?
			      test_grads .template slice<0, 1>(i, q_k) : trial_grads.template slice<0, 1>(i, q_k);
                            fe_packet.test_grad  = test_grads.template slice<0, 1>(j, q_k);
                        }
                        if constexpr (Form::XprBits & fe_assembler_flags::compute_shape_div) {
                            fe_packet.trial_div = is_galerkin ? test_divs(i, q_k) : trial_divs(i, q_k);
                            fe_packet.test_div = test_divs(j, q_k);
                        }
                        if constexpr (Form::XprBits & fe_assembler_flags::compute_physical_quad_nodes) {
                            fe_packet.quad_node_id = local_cell_id * n_quadrature_nodes + q_k;
                        }
                        value += Base::Quadrature::weights[q_k] * form_(fe_packet);
                    }
                    triplet_list.emplace_back(
                      test_active_dofs[j], is_petrov_galerkin ? trial_active_dofs[i] : test_active_dofs[i],
                      value * fe_packet.cell_measure);
                }
            }
	    local_cell_id++;
        }
        return;
    }
    constexpr int n_dofs() const { return trial_dof_handler()->n_dofs(); }
    constexpr int rows() const { return test_dof_handler()->n_dofs(); }
    constexpr int cols() const { return trial_dof_handler()->n_dofs(); }
};

// assembly loop for the discretization of integrals \int_D \langle f, \psi_i \rangle, with \psi_i \in test space
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class vector_fe_linear_form_assembly_loop :
    public vector_fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>,
    public fe_assembly_xpr_base<vector_fe_linear_form_assembly_loop<Triangulation_, Form_, Options_, Quadrature_...>> {
    static constexpr bool trial_space_found = meta::xpr_find<
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; }), std::decay_t<Form_>>();
    fdapde_static_assert(
      !trial_space_found, IF_YOU_WANT_TO_DISCRETIZE_A_BILINEAR_FORM_USE_THE_BILINEAR_FORM_ASSEMBLY_LOOP);
    using Base = vector_fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>;
    using Form = typename Base::Form;
    static constexpr int local_dim = Base::local_dim;
    static constexpr int embed_dim = Base::embed_dim;
    static constexpr int n_basis = Base::n_basis;
    static constexpr int n_quadrature_nodes = Base::n_quadrature_nodes;
    static constexpr int n_components = Base::n_components;
    using Base::dof_handler_;
    using Base::form_;
   public:
    vector_fe_linear_form_assembly_loop() = default;
    vector_fe_linear_form_assembly_loop(
      const Form_& form, typename Base::fe_traits::geo_iterator begin, typename Base::fe_traits::geo_iterator end,
      const Quadrature_&... quadrature) requires(sizeof...(quadrature) <= 1)
        : Base(form, begin, end, quadrature...) { }

    DVector<double> assemble() const {
        DVector<double> assembled_vec(dof_handler_->n_dofs());
        assembled_vec.setZero();
        assemble(assembled_vec);
        return assembled_vec;
    }
    void assemble(DVector<double>& assembled_vec) const {
        using iterator = typename Base::fe_traits::dof_iterator;
        iterator begin(Base::begin_.index(), dof_handler_, Base::begin_.marker());
        iterator end  (Base::end_.index(),   dof_handler_, Base::end_.marker()  );
        // prepare assembly loop
        DVector<int> active_dofs;
        std::unordered_map<const void*, DMatrix<double>> fe_map_buff;   // evaluation of FeMap nodes at quadrature nodes
        if constexpr (Form::XprBits & fe_assembler_flags::compute_physical_quad_nodes) {
            Base::distribute_quadrature_nodes(
              fe_map_buff, begin, end);   // distribute quadrature nodes on physical mesh (if required)
        }
	
        // start assembly loop
        internals::fe_assembler_packet<local_dim> fe_packet(Base::n_components);
	int local_cell_id = 0;
        for (iterator it = begin; it != end; ++it) {
            fe_packet.cell_measure = it->measure();
            active_dofs = it->dofs();
            for (int i = 0; i < n_basis; ++i) {   // test function loop
                double value = 0;
                for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                    // update fe_packet
                    fe_packet.test_value = Base::test_shape_values_.template slice<0, 1>(i, q_k);
                    if constexpr (Form::XprBits & fe_assembler_flags::compute_physical_quad_nodes) {
                        fe_packet.quad_node_id = local_cell_id * n_quadrature_nodes + q_k;
                    }
                    value += Base::Quadrature::weights[q_k] * form_(fe_packet);
                }
                assembled_vec[active_dofs[i]] += value * fe_packet.cell_measure;
            }
	    local_cell_id++;
        }
        return;
    }
    constexpr int n_dofs() const { return dof_handler_->n_dofs(); }
    constexpr int rows() const { return dof_handler_->n_dofs(); }
    constexpr int cols() const { return 1; }
};
  
}   // namespace internals
}   // namespace fdapde

#endif   // __FE_VECTOR_ASSEMBLER_H__
