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

#ifndef __FDAPDE_GEOMETRIC_SCALAR_SPACE_ADAPTER_H__
#define __FDAPDE_GEOMETRIC_SCALAR_SPACE_ADAPTER_H__
#include "header_check.h"

namespace fdapde {
template <typename> class BsSpace;
template <typename, typename> class FeSpace;
template <int, int> struct FeP;
namespace gfe::internals {
/// @brief defines spatial support and basis evaluation for each supported scalar discretization
template <typename Space> struct geometric_scalar_adapter;
/// @brief exposes finite element descriptors only for finite element scalar sources
template <typename Space> struct geometric_space_descriptors { };
/// @brief retains P1 descriptors needed by existing cell quadrature and smoothing adapters
template <typename Mesh> struct geometric_space_descriptors<FeSpace<Mesh, FeP<1, 1>>> {
    using FeType = typename FeSpace<Mesh, FeP<1, 1>>::FeType;
    using cell_dof_descriptor = typename FeType::template cell_dof_descriptor<Mesh::local_dim>;
};
/// @brief reuses scalar P1 numbering and normalized reference shape weights
template <typename Mesh> struct geometric_scalar_adapter<FeSpace<Mesh, FeP<1, 1>>> {
    using Space = FeSpace<Mesh, FeP<1, 1>>;
    using Element = Simplex<Mesh::local_dim, Mesh::embed_dim>;
    using Dofs = std::array<int, Element::n_nodes>;
    using Weights = std::array<double, Element::n_nodes>;
    static constexpr bool is_spline = false;
    /// @brief locates a P1 simplex using the existing spatial search engine
    template <typename Locator, typename Point>
    static int locate(const Space&, const Locator& locator, const Point& x) {
        return locator.locate(x);
    }
    /// @brief retains cell-local scalar indices in the native finite element ordering
    static Dofs dofs(const Space& space, int id) {
        Dofs result;
        for (int i = 0; i < Element::n_nodes; ++i) result[i] = space.dof_handler().dofs()(id, i);
        return result;
    }
    /// @brief evaluates reference weights through the existing P1 scalar basis
    static Weights
    weights(const Space& space, const Element& element, const Dofs&, const typename Element::NodeType& x) {
        return p1_shape_weights(space, element, x);
    }
};
/// @brief supplies active B-spline control indices and physical basis weights on an interval
template <typename Mesh> struct geometric_scalar_adapter<BsSpace<Mesh>> {
    using Space = BsSpace<Mesh>;
    using Element = Simplex<Mesh::local_dim, Mesh::embed_dim>;
    using Dofs = std::vector<int>;
    using Weights = std::vector<double>;
    static constexpr bool is_spline = true;
    /// @brief selects the right-hand knot span at interior knots and the last span at the right endpoint
    template <typename Locator, typename Point> static int locate(const Space& space, const Locator&, const Point& x) {
        const auto& knots = space.triangulation().nodes();
        const auto begin = knots.data(), end = begin + knots.rows();
        if (x[0] < *begin || x[0] > *(end - 1)) return -1;
        if (x[0] == *(end - 1)) return static_cast<int>(knots.rows()) - 2;
        return static_cast<int>(std::upper_bound(begin, end, x[0]) - begin) - 1;
    }
    /// @brief retains the scalar control indices active on one knot span
    static Dofs dofs(const Space& space, int id) { return space.dof_handler().active_dofs(id); }
    /// @brief evaluates nonnegative physical B-spline weights including both clamped endpoints
    static Weights weights(const Space& space, const Element&, const Dofs& dofs, const typename Element::NodeType& x) {
        Weights result;
        result.reserve(dofs.size());
        for (int id : dofs) result.push_back(space.physical_basis()[id](x[0]));
        validate_p1_data(result.size(), result);
        return result;
    }
};
/// @brief accepts only scalar P1 finite elements and interval B-spline spaces
template <typename Space>
concept GeometricScalarSpace = requires { geometric_scalar_adapter<std::remove_cvref_t<Space>>::is_spline; };
}   // namespace gfe::internals
}   // namespace fdapde
#endif
