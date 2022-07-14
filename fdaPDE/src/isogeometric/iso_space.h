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

#ifndef __FDAPDE_ISO_SPACE_H__
#define __FDAPDE_ISO_SPACE_H__

#include "header_check.h"

namespace fdapde {

// forward declarations
template <typename IsoSpace_> class IsoFunction;
namespace internals {

template <typename IsoMesh_, typename Form_, int Options_, typename... Quadrature_>
class iso_bilinear_form_assembly_loop;
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class iso_linear_form_assembly_loop;

}   // namespace internals

template <typename IsoMesh_> class IsoSpace {

    template <typename T> struct subscript_t_impl {
        using type = std::decay_t<decltype(std::declval<T>().operator[](std::declval<int>()))>;
    };
    template <typename T> using subscript_t = typename subscript_t_impl<T>::type;

    public:

    using IsoMesh = std::decay_t<IsoMesh_>;
    static constexpr int local_dim = IsoMesh::local_dim;
    static constexpr int embed_dim = IsoMesh::embed_dim;
    using BasisType = NurbsBasis<local_dim>;
    using ShapeFunctionType = subscript_t<BasisType>;
    using DofHandlerType = DofHandler<local_dim, embed_dim, iso_tag>;
    using discretization_category = iso_tag;
    //static constexpr int sobolev_regularity = 2; ???
    template <typename Triangulation__, typename Form__, int Options__, typename... Quadrature__>
    using bilinear_form_assembly_loop =
      internals::iso_bilinear_form_assembly_loop<Triangulation__, Form__, Options__, Quadrature__...>;
    template <typename Triangulation__, typename Form__, int Options__, typename... Quadrature__>
    using linear_form_assembly_loop =
      internals::iso_linear_form_assembly_loop <Triangulation__, Form__, Options__, Quadrature__...>;

    IsoSpace() = default;

    // constructor with only mesh, uses default basis (same as mesh)
    IsoSpace(const IsoMesh_& mesh) :
        mesh_(std::addressof(mesh)), dof_handler_(mesh), degree_(mesh.basis().degree()), basis_(mesh.basis()) { } 

    // constructor with mesh and basis (customized)
    IsoSpace(const IsoMesh_& mesh, BasisType& basis) :
        mesh_(std::addressof(mesh)), dof_handler_(mesh, basis), degree_(basis.degree()), basis_(basis) { } 

    // observers
    const IsoMesh& mesh() const { return *mesh_; }
    const DofHandlerType& dof_handler() const { return dof_handler_; }
    DofHandlerType& dof_handler() { return dof_handler_; }
    constexpr int n_shape_functions() const { return basis_.size(); }
    //constexpr int n_shape_functions_face() const { return 1; }
    int n_dofs() const { return dof_handler_.n_dofs(); }
    const BasisType& basis() const { return basis_; }
    std::array<int, local_dim> degree() const { return degree_; }


    // evaluations
    template <typename InputType>
        requires(std::is_invocable_v<ShapeFunctionType, InputType>)
    constexpr auto eval_shape_value(int i, const InputType& p) const {
        return basis_[i](p);
    }

    template <typename InputType>
    constexpr auto eval_shape_grad(int i, const InputType& p) const {

        return basis_[i].gradient(p); // da aggiustare
    }

    template <typename InputType>
    constexpr auto eval_shape_hess(int i, const InputType& p) const {
        return basis_[i].hessian(p); // da aggiustare
    }
    

    private:
    
    const IsoMesh* mesh_;
    DofHandlerType dof_handler_;
    BasisType basis_;
    std::array<int, local_dim> degree_;


};




}

#endif
