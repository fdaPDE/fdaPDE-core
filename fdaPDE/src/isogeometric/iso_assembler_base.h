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

#ifndef __FDAPDE_ISO_ASSEMBLER_BASE_H__
#define __FDAPDE_ISO_ASSEMBLER_BASE_H__

#include "header_check.h"

namespace fdapde{

template <typename Derived_> struct IsoMap;

enum class iso_assembler_flags{
    compute_shape_values        = 0x0001,
    compute_shape_grad          = 0x0002,
    compute_shape_hess          = 0x0004,
    compute_shape_div           = 0x0008,
    compute_physical_quad_nodes = 0x0010,
    compute_cell_id             = 0x0020,
    

};

namespace internals {

// information sent from the assembly loop to the integrated forms
template <int EmbedDim> struct iso_assembler_packet {
    //static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;

    iso_assembler_packet() = default;
    iso_assembler_packet(iso_assembler_packet&&) noexcept = default;
    iso_assembler_packet(const iso_assembler_packet&) noexcept = default;

    // geometric informations
    int quad_node_id;       // active physical quadrature node index
    double cell_measure;    // active cell measure
    double cell_id;         // active cell identifier
    double cell_diameter;   // active cell diameter

    // functional informations (Dynamic stands for number of components)
    double trial_value, test_value;            // \psi_i(q_k), \psi_j(q_k)
    MdArray<double, MdExtents<embed_dim>> trial_grad, test_grad;   // \nabla{\psi_i}(q_k), \nabla{\psi_j}(q_k)
    MdArray<double, MdExtents<embed_dim, embed_dim>> trial_hess, test_hess;
};


// base class for vector finite element assembly loops
template<typename IsoMesh_, typename Form_, int Options_, typename... Quadrature_>
struct iso_assembler_base{
    fdapde_static_assert(sizeof...(Quadrature_) < 2, YOU_CAN_SUPPLY_AT_MOST_ONE_QUADRATURE_RULE_TO_A_ISO_ASSEMBLY_LOOP);
    // detect test space (since a test function is always present in a weak form)
    using TestSpace = test_space_t<Form_>;
    using Form =
      std::decay_t<decltype(xpr_wrap<IsoMap, decltype([]<typename Xpr>() {
	    return !(
	        std::is_invocable_v<Xpr, iso_assembler_packet<TestSpace::embed_dim>>);
	  })>(std::declval<Form_>()))>; // vector case ???
    using IsoMesh = typename std::decay_t<IsoMesh_>;
    static constexpr int local_dim = IsoMesh::local_dim;
    static constexpr int embed_dim = IsoMesh::embed_dim;
    static constexpr int Options = Options_;
    using FunctionSpace = TestSpace;
    using DofHandlerType = DofHandler<local_dim, embed_dim, iso_tag>;
    using Quadrature = decltype([]() {
        if constexpr (sizeof...(Quadrature_) == 0) {
            return void();   // quadrature selcted at run-time provided the actual order of spline basis
        } else {
            return std::get<0>(std::tuple<Quadrature_...>());   // user-defined quadrature
        }
    }());
    
    using geo_iterator = std::conditional_t<Options == CellMajor , 
        typename IsoMesh::cell_iterator, typename IsoMesh::boundary_iterator>; 
    
    using dof_iterator = std::conditional_t<Options == CellMajor ,
        typename DofHandlerType::cell_iterator,typename DofHandlerType::boundary_edge_iterator>;  
    using discretization_category = typename TestSpace::discretization_category;
    fdapde_static_assert(
        std::is_same_v<discretization_category FDAPDE_COMMA iso_tag>, THIS_CLASS_IS_FOR_ISO_DISCRETIZATION_ONLY);

    iso_assembler_base() = default;

    iso_assembler_base( const Form_& form, const geo_iterator& begin, const geo_iterator& end, const Quadrature_&... quadrature )
        requires(sizeof...(quadrature)<=1):
        form_(xpr_wrap<IsoMap, decltype([]<typename Xpr>() {
                    return !(std::is_invocable_v<Xpr, iso_assembler_packet<TestSpace::embed_dim>>);
                })>(form)),
        dof_handler_(std::addressof(internals::test_space(form_).dof_handler())),
        test_space_ (std::addressof(internals::test_space(form_))),
        begin_(begin),
        end_(end) { 
            fdapde_assert(dof_handler_->n_dofs() > 0);
            // copy quadrature rule
            Eigen::Matrix<double, Dynamic, Dynamic> quad_nodes__;
            if constexpr (sizeof...(quadrature) == 1) {
                auto quad_rule = std::get<0>(std::make_tuple(quadrature...));
                
                constexpr int int_local_dim = (Options == CellMajor) ? local_dim : local_dim - 1;

                quad_nodes__.resize(quad_rule.order, quad_rule.local_dim);
                quad_weights_.resize(quad_rule.order, 1);
                for (int i = 0; i < quad_rule.order; ++i) {
                    quad_weights_(i, 0) = quad_rule.weights[i];
                    for (int j = 0; j < int_local_dim; ++j) { quad_nodes__(i, j) = quad_rule.nodes(i, j); }
            }
            } else {
                //Using default quadrature rule for iso assembler.
                auto degrees = test_space_->degree();
                int max_degree = *(std::max_element(degrees.begin(), degrees.end()));
                if constexpr(Options == CellMajor){
                    internals::get_iso_quadrature<local_dim>(max_degree, quad_nodes__, quad_weights_);
                } else {
                    internals::get_iso_quadrature<local_dim-1>(max_degree, quad_nodes__, quad_weights_);
                }
            }
            // build grid of quadrature nodes on reference domain
            n_quadrature_nodes_ = quad_nodes__.rows();
            int n_cells = (Options == CellMajor) ? dof_handler_->mesh()->n_cells() : dof_handler_->mesh()->n_boundary_edges() ;
            int n_src_points = quad_nodes__.rows();
            constexpr int int_local_dim = (Options == CellMajor) ? local_dim : local_dim - 1;
            quad_nodes_.resize(n_cells * n_src_points, int_local_dim); // global quad nodes
            int i  = 0;
            int count = 0;
            for(auto it = begin_; it != end_; ++it){
                for(int q_k = 0; q_k < n_quadrature_nodes_; ++q_k){
                    quad_nodes_.row(i) = it->affine_map(quad_nodes__.row(q_k).transpose());
                    i++;
                }
                if(! (Options == CellMajor)) {
                    boundary_ids[it->id()] = count;
                }
                count++;
                
            }
            return;

        }

        const TestSpace& test_space() const { return *test_space_; }
        const IsoMesh& mesh() const { return *dof_handler_->mesh(); }

        protected:
        
        // evaluation of \psi_i(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
        template<typename BasisType__, typename IteratorType, typename DstMdArray>
        void eval_param_shape_values(
            BasisType__&& basis, const std::vector<int>& active_dofs, IteratorType cell, DstMdArray& dst) const {

            using BasisType = std::decay_t<BasisType__>;
            int n_basis =active_dofs.size(); 

            for(int i=0; i < n_basis; ++i){
                // evaluation of \psi_i at q_j, j = 1, ..., n_quadrature_nodes
                for(int j=0; j < n_quadrature_nodes_; ++j){

                    if constexpr (Options == CellMajor) {
                        dst(i, j) = basis[active_dofs[i]](quad_nodes_.row(cell->id() * n_quadrature_nodes_ + j).transpose());}
                    else {
                        //for boundary cells we need to build the param_point
                        int id = boundary_ids.at(cell->id());
                        dst(i, j) = basis[active_dofs[i]](cell->param_point(quad_nodes_.row(id * n_quadrature_nodes_ + j).transpose()));
                    }

                                    
                }
            }
            return;
        }
        

        // evaluation of 1-st order derivative of basis function
        template <typename BasisType__, typename IteratorType, typename DstMdArray>
            void eval_param_shape_grads(
            BasisType__&& basis, const std::vector<int>& active_dofs, IteratorType cell, DstMdArray& dst) const {
            using BasisType = std::decay_t<BasisType__>;
            int n_basis = active_dofs.size();
            for (int i = 0; i < n_basis; ++i) {
                for (int j = 0; j < n_quadrature_nodes_; ++j) {  
                    auto der = basis[active_dofs[i]].gradient(quad_nodes_.row(cell->id() * n_quadrature_nodes_ + j).transpose());
                    for(int k = 0; k < local_dim; ++k){
                        dst(i, j, k) = der(k);
                    }
                }
            }
            return;
        }

        //evaluation of hessian matrix of \psi_i(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
        template <typename BasisType__, typename IteratorType, typename DstMdArray>
        void eval_param_shape_hess(
            BasisType__&& basis, const std::vector<int>& active_dofs, IteratorType cell, DstMdArray& dst) const {
            
            using BasisType = std::decay_t<BasisType__>;
            int n_basis = active_dofs.size();
        
            for (int i = 0; i < n_basis; ++i) {
                for (int j = 0; j < n_quadrature_nodes_; ++j) {
                    auto qp = quad_nodes_.row(cell->id() * n_quadrature_nodes_ + j).transpose();
                    auto hess = basis[active_dofs[i]].hessian(qp);  // Assuming this returns a matrix-like object
        
                    for (int k = 0; k < local_dim; ++k) {
                        for (int l = 0; l < local_dim; ++l) {
                            dst(i, j, k, l) = hess(k, l);
                        }
                    }
                }
            }
        }
        template <typename IteratorType, typename DstMdArray>
        void eval_metric_determinant(IteratorType cell, DstMdArray& dst) const {
            // evaluation of metric determinant on quadrature nodes
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                if constexpr(Options == CellMajor) {
                    dst(j) = cell->metric_determinant(quad_nodes_.row(cell->id() * n_quadrature_nodes_ + j).transpose(),true);
                } else {
                    // for boundary cells we need to build the param_point and use a different id
                    int id = boundary_ids.at(cell->id());
                    dst(j) = cell->metric_determinant(quad_nodes_.row(id * n_quadrature_nodes_ + j).transpose());
                }
            }
        }

        // eval_param 
        template <typename IteratorType, typename DstMdArray>
        void eval_param_grad(IteratorType cell, DstMdArray& dst) const {
            // evaluation of parametric gradient on quadrature nodes
            // dst is a member of the packet
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                // evaluation of metric determinant on q_j, j = 1, ..., n_quadrature_nodes
                dst(j) = cell->parametrization_gradient(quad_nodes_.row(cell->id() * n_quadrature_nodes_ + j),true);
            }
        }


        template <typename IteratorType, typename DstMdArray>
        void eval_param_hess(IteratorType cell, DstMdArray& dst) const {
            // evaluation of parametric gradient on quadrature nodes
            // dst is a member of the packet
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                // evaluation of metric determinant on q_j, j = 1, ..., n_quadrature_nodes
                dst(j) = cell->parametrization_hessian(quad_nodes_.row(cell->id() * n_quadrature_nodes_ + j),true);
            }
        }


        void distribute_quadrature_nodes (
             dof_iterator begin,
            dof_iterator end) const {
              Eigen::Matrix<double, Dynamic, Dynamic> phys_quad_nodes;
              // not need the linear map since they are already in the parametric space
              
              phys_quad_nodes.resize(n_quadrature_nodes_ * (end_.index() - begin_.index()), embed_dim);
              int local_cell_id = 0;
              for (geo_iterator it = begin_; it != end_; ++it) {
                  for (int q_k = 0; q_k < n_quadrature_nodes_; ++q_k) {
                      phys_quad_nodes.row(local_cell_id * n_quadrature_nodes_ + q_k) =
                        it->parametrization(quad_nodes_.row(local_cell_id * n_quadrature_nodes_ + q_k).transpose(), true);
                  }
                  local_cell_id++;
              }
              
              // evaluate Map nodes at quadrature nodes
              xpr_apply_if<
                decltype([]<typename Xpr_, typename... Args>(Xpr_& xpr, Args&&... args) {
                    xpr.init(std::forward<Args>(args)...);
                    return;
                }),
                decltype([]<typename Xpr_>() {
                    return requires(Xpr_ xpr) { xpr.init(phys_quad_nodes, begin, end); };
                })>(form_, phys_quad_nodes, begin, end);
              return;
          }


    protected:
    Form form_;
    const DofHandlerType* dof_handler_;
    const TestSpace* test_space_;
    geo_iterator begin_, end_;
    // quadrature
    Eigen::Matrix<double, Dynamic, Dynamic> quad_nodes_, quad_weights_;
    int n_quadrature_nodes_;
    std::map<int,int> boundary_ids;





};




}






}





#endif