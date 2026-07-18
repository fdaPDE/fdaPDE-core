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

#ifndef __FDAPDE_FE_LINEAR_FORM_ASSEMBLER_H__
#define __FDAPDE_FE_LINEAR_FORM_ASSEMBLER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {
  
// assembly loop for the discretization of integrals \int_D \langle f, \psi_i \rangle, with \psi_i \in test space
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class fe_linear_form_assembly_loop :
    public fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>,
    public assembly_xpr_base<fe_linear_form_assembly_loop<Triangulation_, Form_, Options_, Quadrature_...>> {
   public:
    using Base = fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>;
    using Form = typename Base::Form;
    using discretization_category = typename Base::discretization_category;
    static constexpr int local_dim = Base::local_dim;
    static constexpr int embed_dim = Base::embed_dim;
    static constexpr int n_basis = Base::n_basis;
    static constexpr int n_quadrature_nodes = Base::n_quadrature_nodes;
    static constexpr int n_components = Base::n_components;
    using Base::dof_handler_;
    using Base::form_;

   private:
    void assemble_interior_facets(Eigen::Matrix<double, Dynamic, 1>& assembled_vec) const {
        fdapde_static_assert(Options_ == CellMajor, INTERIOR_FACET_TERMS_REQUIRE_A_CELL_MAJOR_ASSEMBLY_LOOP);
        fdapde_static_assert(
          local_dim == 2 && embed_dim == 2, INTERIOR_FACET_ASSEMBLY_IS_CURRENTLY_IMPLEMENTED_FOR_PLANAR_TRIANGLES);
        fdapde_static_assert(
          n_components == 1, INTERIOR_FACET_ASSEMBLY_IS_CURRENTLY_IMPLEMENTED_FOR_SCALAR_FINITE_ELEMENTS);
        fdapde_static_assert(
          !(Form::XprBits & int(fe_assembler_flags::compute_shape_div)),
          INTERIOR_FACET_DIVERGENCE_TERMS_ARE_NOT_IMPLEMENTED);

        using FacetQuadrature = typename Base::FeType::template face_quadrature_t<local_dim>;
        constexpr int n_facet_quadrature_nodes = FacetQuadrature::order;

        const auto& space = Base::test_space();
        const auto& mesh = space.triangulation();
        if (Base::begin_.marker() != TriangulationAll || Base::begin_.index() != 0 ||
            Base::end_.index() != mesh.n_cells()) {
            throw std::invalid_argument("DG interior-facet assembly currently requires the complete mesh");
        }

        internals::fe_assembler_packet<embed_dim> fe_packet(n_components);
        fe_packet.interior_facet = true;
        const std::array<fe_facet_side, 2> facet_sides {fe_facet_side::plus, fe_facet_side::minus};

        for (auto edge = mesh.edges_begin(); edge != mesh.edges_end(); ++edge) {
            if (edge->on_boundary()) continue;
            auto adjacent_cells = edge->adjacent_cells();
            std::array<int, 2> cell_ids {
              std::min(adjacent_cells[0], adjacent_cells[1]), std::max(adjacent_cells[0], adjacent_cells[1])};
            std::array<typename Base::DofHandlerType::CellType, 2> cells {
              dof_handler_->cell(cell_ids[0]), dof_handler_->cell(cell_ids[1])};
            std::array<Eigen::Matrix<int, Dynamic, 1>, 2> test_dofs {
              dof_handler_->active_dofs(cell_ids[0]), dof_handler_->active_dofs(cell_ids[1])};

            Eigen::Matrix<double, embed_dim, 1> tangent = edge->node(1) - edge->node(0);
            Eigen::Matrix<double, embed_dim, 1> normal;
            normal << -tangent[1], tangent[0];
            normal.normalize();
            if (normal.dot(cells[1].barycenter() - cells[0].barycenter()) < 0) normal = -normal;

            fe_packet.geo_id = edge->id();
            fe_packet.measure = edge->measure();
            fe_packet.facet_size = edge->measure();
            fe_packet.normal.assign_inplace_from(normal.data());
            fe_packet.facet_normal.assign_inplace_from(normal.data());

            for (int test_side = 0; test_side < 2; ++test_side) {
                fe_packet.test_side = facet_sides[test_side];
                for (int j = 0; j < n_basis; ++j) {
                    double value = 0;
                    for (int q_k = 0; q_k < n_facet_quadrature_nodes; ++q_k) {
                        Eigen::Matrix<double, embed_dim, 1> point =
                          edge->node(0) + edge->J().col(0) * FacetQuadrature::nodes[q_k];
                        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_physical_quad_nodes)) {
                            fe_packet.physical_quad_node.assign_inplace_from(point.data());
                        }
                        Eigen::Matrix<double, local_dim, 1> test_ref_point =
                          cells[test_side].invJ() * (point - cells[test_side].node(0));

                        fe_packet.test_value[0] = space.eval_shape_value(j, test_ref_point);
                        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_grad)) {
                            auto test_grad = space.eval_cell_grad(j, cell_ids[test_side], test_ref_point);
                            fe_packet.test_grad.assign_inplace_from(test_grad.data());
                        }
                        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_hess)) {
                            auto test_hess = space.eval_cell_hess(j, cell_ids[test_side], test_ref_point);
                            fe_packet.test_hess.assign_inplace_from(test_hess.data());
                        }
                        fe_packet.trace_side = fe_facet_side::none;
                        value += FacetQuadrature::weights[q_k] * form_(fe_packet);
                    }
                    assembled_vec[test_dofs[test_side][j]] += value * edge->measure();
                }
            }
        }
    }

   public:
  
    fe_linear_form_assembly_loop() = default;
    fe_linear_form_assembly_loop(
      const Form_& form, typename Base::fe_traits::geo_iterator begin, typename Base::fe_traits::geo_iterator end,
      const Quadrature_&... quadrature) requires(sizeof...(quadrature) <= 1)
        : Base(form, begin, end, quadrature...) { }

    Eigen::Matrix<double, Dynamic, 1> assemble() const {
        Eigen::Matrix<double, Dynamic, 1> assembled_vec(dof_handler_->n_dofs());
        assembled_vec.setZero();
        assemble(assembled_vec);
        return assembled_vec;
    }
    void assemble(Eigen::Matrix<double, Dynamic, 1>& assembled_vec) const {
        using iterator = typename Base::fe_traits::dof_iterator;
        iterator begin(Base::begin_.index(), dof_handler_, Base::begin_.marker());
        iterator end  (Base::end_.index(),   dof_handler_, Base::end_.marker()  );
        // prepare assembly loop
        Eigen::Matrix<int, Dynamic, 1> active_dofs;
        MdArray<double, MdExtents<n_basis, n_quadrature_nodes, embed_dim, n_components>> test_grads;

        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_physical_quad_nodes)) {
            Base::distribute_quadrature_nodes(begin, end);
        }
        // start assembly loop
        internals::fe_assembler_packet<embed_dim> fe_packet(Base::n_components);
	int local_cell_id = 0;
        for (iterator it = begin; it != end; ++it) {
            fe_packet.measure = it->measure();
            if constexpr (Form::XprBits & int(geo_assembler_flags::compute_geo_id)) { fe_packet.geo_id = it->id(); }
            if constexpr (Form::XprBits & int(geo_assembler_flags::compute_face_normal)) {
                fdapde_static_assert(Options_ == FaceMajor, LINEAR_FORM_REQUIRES_A_FACE_MAJOR_ASSEMBLY_LOOP);
                fe_packet.normal = it->normal();
            }
            if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_grad)) {
                Base::eval_shape_grads_on_cell(it, Base::test_shape_grads_, test_grads);
            }

            // perform integration of linear form for i-th basis
            active_dofs = it->dofs();
            for (int i = 0; i < n_basis; ++i) {   // test function loop
                double value = 0;
                for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                    // update fe_packet
                    fe_packet.test_value.assign_inplace_from(Base::test_shape_values_.template slice<0, 1>(i, q_k));
                    if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_grad)) {
                        fe_packet.test_grad.assign_inplace_from(test_grads.template slice<0, 1>(i, q_k));
                    }
                    if constexpr (Form::XprBits & int(fe_assembler_flags::compute_physical_quad_nodes)) {
                        fe_packet.quad_node_id = local_cell_id * n_quadrature_nodes + q_k;
                    }
                    value += Base::Quadrature::weights[q_k] * form_(fe_packet);
                }
                assembled_vec[active_dofs[i]] += value * fe_packet.measure;
            }
	    local_cell_id++;
        }
        if constexpr (Form::XprBits & int(fe_assembler_flags::interior_facet)) {
            assemble_interior_facets(assembled_vec);
        }
        return;
    }
    constexpr int n_dofs() const { return dof_handler_->n_dofs(); }
    constexpr int rows() const { return n_dofs(); }
    constexpr int cols() const { return 1; }
};
  
}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_FE_LINEAR_FORM_ASSEMBLER_H__
