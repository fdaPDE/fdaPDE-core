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

#ifndef __FDAPDE_BS_SPACE_H__
#define __FDAPDE_BS_SPACE_H__

#include "header_check.h"

namespace fdapde {

// forward declarations
template <typename SpSpace_> class SpFunction;
namespace internals {

template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class sp_bilinear_form_assembly_loop;
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class sp_linear_form_assembly_loop;

}   // namespace internals
  
template <typename Triangulation_> class BsSpace {
    fdapde_static_assert(
      Triangulation_::local_dim == 1 && Triangulation_::embed_dim == 1, THIS_CLASS_IS_FOR_INTERVAL_MESHES_ONLY);
    template <typename T> struct subscript_t_impl {
        using type = std::decay_t<decltype(std::declval<T>().operator[](std::declval<int>()))>;
    };
    template <typename T> using subscript_t = typename subscript_t_impl<T>::type;
   public:
    using Triangulation = std::decay_t<Triangulation_>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    using BasisType = BSplineBasis;
    using ShapeFunctionType = subscript_t<BasisType>;
    using DofHandlerType = DofHandler<local_dim, embed_dim, spline_tag>;
    using discretization_category = spline_tag;
    constexpr int sobolev_regularity() const { return 2; }
    template <typename Triangulation__, typename Form__, int Options__, typename... Quadrature__>
    using bilinear_form_assembly_loop =
      internals::sp_bilinear_form_assembly_loop<Triangulation__, Form__, Options__, Quadrature__...>;
    template <typename Triangulation__, typename Form__, int Options__, typename... Quadrature__>
    using linear_form_assembly_loop =
      internals::sp_linear_form_assembly_loop  <Triangulation__, Form__, Options__, Quadrature__...>;

    BsSpace() = default;
    BsSpace(const Triangulation_& interval, int order) :
        triangulation_(std::addressof(interval)),
        dof_handler_(interval),
        physical_basis_(interval, order),
        order_(order) {
        a_ = triangulation_->bbox()[0], b_ = triangulation_->bbox()[1];   // store interval range
        dof_handler_.enumerate(BasisType(interval, order));
	// build reference [-1, 1] interval with nodes mapped from physical interval [a, b]
        Eigen::Matrix<double, Dynamic, 1> ref_nodes(triangulation_->n_nodes());
        for (int i = 0; i < triangulation_->n_nodes(); ++i) {
            ref_nodes[i] = map_to_reference(triangulation_->nodes()(i, 0));
        }
        // generate basis on reference [-1, 1] interval
        basis_ = BasisType(Triangulation(ref_nodes), order);
    }
    // observers
    const Triangulation& triangulation() const { return *triangulation_; }
    const DofHandlerType& dof_handler() const { return dof_handler_; }
    DofHandlerType& dof_handler() { return dof_handler_; }
    constexpr int n_shape_functions() const { return basis_.size(); }
    constexpr int n_shape_functions_face() const { return 1; }
    int n_dofs() const { return dof_handler_.n_dofs(); }
    const BasisType& basis() const { return basis_; }
    const BasisType& physical_basis() const { return physical_basis_; }
    int order() const { return order_; }
    // evaluation
    template <typename InputType>
        requires(std::is_invocable_v<ShapeFunctionType, InputType>)
    constexpr auto eval_shape_value(int i, const InputType& p) const {
        return basis_[i](p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<ShapeFunctionType>().gradient(1)), InputType>)
    constexpr auto eval_shape_dx(int i, const InputType& p) const {
        return basis_[i].gradient(1)(p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<ShapeFunctionType>().gradient(2)), InputType>)
    constexpr auto eval_shape_ddx(int i, const InputType& p) const {
        return basis_[i].gradient(2)(p);
    }
    template <typename InputType>
    constexpr auto eval_face_shape_value(int i, [[maybe_unused]] const InputType& p) const {
        return (i == 0 || i == n_dofs() - 1) ? 1.0 : 0.0;
    }
    template <typename InputType> constexpr auto eval_face_shape_dx (int i, const InputType& p) const {
        return basis_[i].gradient(1)(p);
    }
    template <typename InputType> constexpr auto eval_face_shape_ddx(int i, const InputType& p) const {
        return basis_[i].gradient(2)(p);
    }
    // evaluation on physical domain
    // evaluate value of the i-th shape function defined on physical domain [a, b]
    template <typename InputType> auto eval_cell_value(int i, [[maybe_unused]] int cell_id, const InputType& p) const {
        return eval_cell_value(i, p);   // cell_id unused since there is no point location involved
    }
    template <typename InputType> auto eval_cell_value(int i, const InputType& p) const {
        double p_;
        if constexpr (internals::is_subscriptable<InputType, int>) {
            p_ = p[0];
        } else {
            p_ = p;
        }
        if (p_ < triangulation_->bbox()[0] || p_ > triangulation_->bbox()[1]) {
            return std::numeric_limits<double>::quiet_NaN();   // return NaN if point lies outside domain
        }
        return eval_shape_value(i, Matrix<double, embed_dim, 1>(map_to_reference(p_)));
    }
    // return i-th basis function on physical domain
    SpFunction<BsSpace<Triangulation_>> operator[](int i) {
        fdapde_assert(i < dof_handler_.n_dofs());
	Eigen::Matrix<double, Dynamic, 1> coeff = Eigen::Matrix<double, Dynamic, 1>::Zero(dof_handler_.n_dofs());
	coeff[i] = 1;
        return BsFunction<BsSpace<Triangulation_>>(*this, coeff);
    }
   private:
    double a_, b_;   // triangulation range
    // linear mapping from p \in [a, b] to \hat p \in [-1, +1]
    double map_to_reference(double p) const { return 2 * (p - a_) / (b_ - a_) - 1; }

    const Triangulation* triangulation_;
    DofHandlerType dof_handler_;   // dof_handler over physical domain
    BasisType physical_basis_;     // basis over physical interval [a, b]
    BasisType basis_;              // basis_ over reference interval [-1, +1]
    int order_;                    // spline order
};

}   // namespace fdapde

#endif   // __FDAPDE_BS_SPACE_H__
