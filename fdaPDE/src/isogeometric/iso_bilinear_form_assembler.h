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

#ifndef __FDAPDE_ISO_BILINEAR_FORM_ASSEMBLER_H__
#define __FDAPDE_ISO_BILINEAR_FORM_ASSEMBLER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

template<typename IsoMesh_, typename Form_, int Options_, typename... Quadrature_>
class iso_bilinear_form_assembly_loop :
    public iso_assembler_base<IsoMesh_, Form_, Options_, Quadrature_...>,
    public assembly_xpr_base<iso_bilinear_form_assembly_loop<IsoMesh_, Form_, Options_, Quadrature_...>> {
    // detect trial and test spaces from bilinear form
    public:
    using TestSpace = test_space_t<Form_>;
    using TrialSpace = trial_space_t<Form_>;
    static_assert(TestSpace::local_dim == TrialSpace::local_dim && TestSpace::embed_dim == TrialSpace::embed_dim);
    static constexpr bool is_galerkin = std::is_same_v<TestSpace, TrialSpace>;
    static constexpr bool is_petrov_galerkin = !is_galerkin;
    using Base = iso_assembler_base<IsoMesh_, Form_, Options_, Quadrature_...>;
    using Form = typename Base::Form;
    
    using DofHandlerType = typename Base::DofHandlerType;
    using discretization_category = typename TestSpace::discretization_category;
    private:

    fdapde_static_assert(
        std::is_same_v<discretization_category FDAPDE_COMMA iso_tag>, TEST_AND_TRIAL_SPACE_MUST_HAVE_THE_SAME_DISCRETIZATION_CATEGORY);
    static constexpr int local_dim = Base::local_dim;
    static constexpr int embed_dim = Base::embed_dim;
    using Base::form_;
    using Base::test_space_;
    // private data members
    const DofHandlerType* trial_dof_handler_;
    constexpr const DofHandlerType* test_dof_handler() const { return Base::dof_handler_; }
    constexpr const DofHandlerType* trial_dof_handler() const {
        return is_galerkin ? Base::dof_handler_ : trial_dof_handler_;
    }
    const TrialSpace* trial_space_;

    public:

    inline int map_dof(int i, const std::vector<int>& map) {
        return map.empty() ? i : map[i];
    }
    
    iso_bilinear_form_assembly_loop() = default;
    
    iso_bilinear_form_assembly_loop(
        const Form_& form, typename Base::geo_iterator begin, typename Base::geo_iterator end, const Quadrature_&... quadrature)
        requires(sizeof...(quadrature)<=1) 
        : Base(form, begin, end, quadrature...), trial_space_(std::addressof(internals::trial_space(form_))){
            if constexpr(is_petrov_galerkin){
                trial_dof_handler_ = std::addressof(internals::trial_space(form_).dof_handler());
            }
            fdapde_assert(test_dof_handler()->n_dofs() != 0 && trial_dof_handler()->n_dofs() != 0);
            
            /* petrov galerking not supported yet
            if constexpr (sizeof...(Quadrature_) == 0) {
                // default to higher-order quadrature
                if (test_space_->degree() != trial_space_->degree()) {
                    internals::get_iso_quadrature<local_dim>(
                    test_space_->degree() > trial_space_->degree() ? test_space_->degree() : trial_space_->degree(),
                    Base::quad_nodes_, Base::quad_weights_);
                }
            }
            */
    }


    Eigen::SparseMatrix<double> assemble() const {
        Eigen::SparseMatrix<double> assembled_mat(test_dof_handler()->n_dofs(), trial_dof_handler()->n_dofs());
        std::vector<Eigen::Triplet<double>> triplet_list;
        assemble(triplet_list);
        assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
        assembled_mat.makeCompressed();
        return assembled_mat;
    }


    void assemble(std::vector<Eigen::Triplet<double>>& triplet_list) const{
        using iterator = typename Base::dof_iterator;
        iterator begin(Base::begin_.index(), test_dof_handler(), Base::begin_.marker());
        iterator end  (Base::end_.index()  , test_dof_handler(), Base::end_.marker());

        // prepare assembly loop
        std::vector<int> test_active_dofs, trial_active_dofs;
        int q = Base::n_quadrature_nodes_;
        int n1 = 1, n2 = 1;

        for(int i = 0; i<local_dim; i++){
            n1*= (test_space_->degree())[i] + 1;
            n2*= (is_galerkin ? (test_space_->degree())[i] : (trial_space_->degree())[i]) + 1;
        }

        MdArray<double, MdExtents<Dynamic, Dynamic>> test_param_shape_values(n1,q), trial_param_shape_values(n2, q);
        MdArray<double, MdExtents<Dynamic, Dynamic, Dynamic>> 
            test_param_shape_grads(n1, q, local_dim), trial_param_shape_grads(n2, q, local_dim);
        MdArray<double, MdExtents<Dynamic, Dynamic, Dynamic, Dynamic>> 
            test_param_shape_hess(n1, q, local_dim, local_dim), trial_param_shape_hess(n2, q, local_dim, local_dim);

        MdArray<Eigen::Matrix<double, embed_dim, local_dim> , MdExtents< Dynamic>> 
            param_grad(q);

        MdArray<MdArray<double, MdExtents<embed_dim, local_dim, local_dim>>, MdExtents<Dynamic>> 
            param_hess(q);

        MdArray<Eigen::Matrix<double, embed_dim, local_dim> , MdExtents< Dynamic>> grad_transf(q);
        
        MdArray<double, MdExtents<Dynamic>> metric_dets(q);

        if constexpr (Form::XprBits & int(iso_assembler_flags::compute_physical_quad_nodes)) {
            Base::distribute_quadrature_nodes(begin, end);
        }

        

        // start assembly loop
        internals::iso_assembler_packet<embed_dim> iso_packet {};
        int local_cell_id = 0;

        

        for(iterator it = begin; it!= end; ++it) {

            test_active_dofs = it->dofs();

            if constexpr (is_petrov_galerkin) { trial_active_dofs = trial_dof_handler()->active_dofs(it->id()); } // petrov galerkin not supported yet
            // update the iso_packet
            iso_packet.cell_measure = it->parametric_measure();

            if constexpr (Form::XprBits & int(iso_assembler_flags::compute_shape_values)) {
                Base::eval_param_shape_values(test_space_->basis(), test_active_dofs, it, test_param_shape_values);
                Base::eval_param_shape_values(
                    trial_space_->basis(), is_petrov_galerkin ? trial_active_dofs : test_active_dofs, it,
                    trial_param_shape_values);
                Base::eval_metric_determinant(it, metric_dets);
            }

            if constexpr (Form::XprBits & int(iso_assembler_flags::compute_shape_grad)) {
                //std::cout<<"Computing grads..."<<std::endl;
                Base::eval_param_shape_grads(test_space_->basis(), test_active_dofs, it, test_param_shape_grads);
                Base::eval_param_shape_grads(
                  trial_space_->basis(), is_petrov_galerkin ? trial_active_dofs : test_active_dofs, it, trial_param_shape_grads);

                    // precompute F(F^T F)^-1 for each q_k
                    Base::eval_param_grad(it, param_grad); // F
                    for (int q_k = 0; q_k < Base::n_quadrature_nodes_; ++q_k) {
                        auto G = param_grad(q_k).transpose() * param_grad(q_k);
                        grad_transf(q_k) = param_grad(q_k) * G.inverse();
                        metric_dets(q_k) = std::sqrt(G.determinant());
                    }
                    
                
            }

            if constexpr (Form::XprBits & int(iso_assembler_flags::compute_shape_hess)) { 
                Base::eval_param_shape_grads(test_space_->basis(), test_active_dofs, it, test_param_shape_grads);
                Base::eval_param_shape_grads(
                  trial_space_->basis(), is_petrov_galerkin ? trial_active_dofs : test_active_dofs, it, trial_param_shape_grads);
                Base::eval_param_shape_hess(test_space_->basis(), test_active_dofs, it, test_param_shape_hess);
                Base::eval_param_shape_hess(
                  trial_space_->basis(), is_petrov_galerkin ? trial_active_dofs : test_active_dofs, it,
                  trial_param_shape_hess);
                Base::eval_param_hess(it, param_hess);
                Base::eval_param_grad(it, param_grad); // F
                for (int q_k = 0; q_k < Base::n_quadrature_nodes_; ++q_k) {
                    auto G = param_grad(q_k).transpose() * param_grad(q_k);
                    grad_transf(q_k) = param_grad(q_k) * G.inverse();
                    metric_dets(q_k) = std::sqrt(G.determinant());
                }

            }

            //Base::eval_metric_determinant(it, metric_dets); // metric sqrt det(F^T F) gia fatto sopra


            // perform integration of weak form for (i,j)-th basis pair
            for(int i = 0; i<n2; ++i){
                for(int j = 0; j<n1; ++j){
                    double value = 0;
                    for (int q_k = 0; q_k < Base::n_quadrature_nodes_; ++q_k) {
                        if constexpr (Form::XprBits & int(iso_assembler_flags::compute_shape_values)) {
                            iso_packet.trial_value = trial_param_shape_values(i, q_k) ;
                            iso_packet.test_value  = test_param_shape_values (j, q_k) ;


                        }
                        if constexpr (Form::XprBits & int(iso_assembler_flags::compute_shape_grad)) {
                            auto temp_trial_grad = trial_param_shape_grads.template slice<0,1>(i, q_k); 
                            auto temp_test_grad  = test_param_shape_grads.template slice<0,1>(j, q_k);
                            Eigen::Matrix<double, local_dim, 1> trial_grad, test_grad;

                            for(int k = 0; k < local_dim; ++k) {
                                trial_grad(k) =  temp_trial_grad(k);
                                test_grad(k)  = temp_test_grad(k);
                            }

                            Eigen::Matrix<double, embed_dim, 1> phys_trial_grad = grad_transf(q_k) *  trial_grad;
                            Eigen::Matrix<double, embed_dim, 1> phys_test_grad  = grad_transf(q_k) * test_grad;

                            for(int k = 0; k < embed_dim; ++k) {
                                iso_packet.trial_grad(k) = phys_trial_grad(k);
                                iso_packet.test_grad(k)  = phys_test_grad(k);
                            }


                        }

                        if constexpr (Form::XprBits & int(iso_assembler_flags::compute_shape_hess)) {
                            auto temp_trial_grad = trial_param_shape_grads.template slice<0,1>(i, q_k); 
                            auto temp_test_grad  = test_param_shape_grads.template slice<0,1>(j, q_k);
                            Eigen::Matrix<double, local_dim, 1> trial_grad, test_grad;
                            for(int k = 0; k < local_dim; ++k) {
                                trial_grad(k) =  temp_trial_grad(k);
                                test_grad(k)  = temp_test_grad(k);
                            }
                            auto param_trial_hess = trial_param_shape_hess.template slice<0,1>(i, q_k);
                            auto param_test_hess  = test_param_shape_hess.template slice<0,1>(j, q_k);
                            Eigen::Matrix<double, local_dim, local_dim> trial_hess, test_hess;
                            for(int k = 0; k < local_dim; ++k) {
                                for(int l = 0; l < local_dim; ++l) {
                                    trial_hess(k,l) = param_trial_hess(k,l);
                                    test_hess(k,l)  = param_test_hess(k,l);
                                }
                            }
                            
                            // --- START curvature-corrected Hessian logic ---
                            // Compute dξ/dx (inverse Jacobian)
                            Eigen::Matrix<double, local_dim, embed_dim> dxi_dx = (param_grad(q_k).transpose() * param_grad(q_k)).inverse() * param_grad(q_k).transpose();
                            // Compute pullback Hessian
                            Eigen::Matrix<double, embed_dim, embed_dim> phys_trial_hess;
                            Eigen::Matrix<double, embed_dim, embed_dim> phys_test_hess;
                            Eigen::Matrix<double, embed_dim, embed_dim> P;
                            // Project onto the tangent plane
                            if constexpr(embed_dim == 2){
                                 P = Eigen::Matrix<double, embed_dim, embed_dim>::Identity();
                            }
                            else{
                                Eigen::Matrix<double, embed_dim, 1> n = ((param_grad(q_k).col(0)).cross(param_grad(q_k).col(1))).normalized();
                                P = Eigen::Matrix<double, embed_dim, embed_dim>::Identity() - n * n.transpose();
                            }
                            
                            phys_trial_hess.setZero();
                            phys_test_hess.setZero();
                            phys_trial_hess = dxi_dx.transpose() * trial_hess * dxi_dx;
                            phys_test_hess  = dxi_dx.transpose() * test_hess  * dxi_dx;
                            // Add curvature correction term
                            for (int ii = 0; ii < embed_dim; ++ii) {
                                for (int jj = 0; jj < embed_dim; ++jj) {
                                    for (int alpha = 0; alpha < local_dim; ++alpha) {
                                        double d2xi = 0.0;
                                        for (int kk = 0; kk < embed_dim; ++kk) {
                                            for (int beta = 0; beta < local_dim; ++beta) {
                                                for (int gamma = 0; gamma < local_dim; ++gamma) {
                                                    d2xi -= dxi_dx(alpha, kk) * param_hess(q_k)(kk, beta, gamma) * dxi_dx(beta, ii) * dxi_dx(gamma, jj);                
                                                }
                                            }
                                        }
                                        phys_trial_hess(ii, jj) += trial_grad(alpha) * d2xi;
                                        phys_test_hess(ii, jj)  += test_grad(alpha)  * d2xi;
                                    }
                                }
                            }
                                

                            auto phys_trial_hess_ = P * phys_trial_hess * P; //P * phys_trial_hess * P;
                            auto phys_test_hess_  = P * phys_test_hess * P; ; //P * phys_test_hess * P;
                            
                            for(int k = 0; k < embed_dim; ++k) {
                                for(int l = 0; l < embed_dim; ++l) {
                                    iso_packet.trial_hess(k,l) = phys_trial_hess_(k,l) ;
                                    iso_packet.test_hess(k,l)  = phys_test_hess_(k,l) ;
                                }
                            }
                        }
 
                        if constexpr (Form::XprBits & int(iso_assembler_flags::compute_physical_quad_nodes)) {
                            iso_packet.quad_node_id = local_cell_id * Base::n_quadrature_nodes_ + q_k;
                        }
                        auto eval_form = form_(iso_packet);
                        value += Base::quad_weights_(q_k, 0) * eval_form * metric_dets(q_k);
                    }

                    triplet_list.emplace_back(
                        test_active_dofs[j],
                        is_galerkin ? test_active_dofs[i] : trial_active_dofs[i],
                        value * iso_packet.cell_measure);
                }   


            }
            local_cell_id++;
        };

        return;

    }

    constexpr int n_dofs() const { return trial_dof_handler()->n_dofs(); }
    constexpr int rows() const { return test_dof_handler()->n_dofs(); }
    constexpr int cols() const { return trial_dof_handler()->n_dofs(); }
    constexpr const TrialSpace& trial_space() const { return *trial_space_; } 

    
};



} // namespace internals
} // namespace fdapde

#endif // __FDAPDE_ISO_BILINEAR_FORM_ASSEMBLER_H__