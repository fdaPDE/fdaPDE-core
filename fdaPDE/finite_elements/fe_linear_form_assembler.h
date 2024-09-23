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

#ifndef __FE_LINEAR_FORM_ASSEMBLER_H__
#define __FE_LINEAR_FORM_ASSEMBLER_H__

#include <unordered_map>

#include "fe_assembler_base.h"

namespace fdapde {
namespace internals {
  
// assembly loop for the discretization of integrals \int_D \langle f, \psi_i \rangle, with \psi_i \in test space
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class fe_linear_form_assembly_loop :
    public fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>,
    public fe_assembly_xpr_base<fe_linear_form_assembly_loop<Triangulation_, Form_, Options_, Quadrature_...>> {
    using Base = fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>;
    using Form = typename Base::Form;
    static constexpr int local_dim = Base::local_dim;
    static constexpr int embed_dim = Base::embed_dim;
    static constexpr int n_basis = Base::n_basis;
    static constexpr int n_quadrature_nodes = Base::n_quadrature_nodes;
    static constexpr int n_components = Base::n_components;
    using Base::dof_handler_;
    using Base::form_;
   public:
    fe_linear_form_assembly_loop() = default;
    fe_linear_form_assembly_loop(
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
                    fe_packet.test_value.assign_inplace_from(Base::test_shape_values_.template slice<0, 1>(i, q_k));
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

#endif   // __FE_LINEAR_FORM_ASSEMBLER_H__
