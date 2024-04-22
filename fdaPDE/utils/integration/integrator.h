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

#ifndef __INTEGRATOR_H__
#define __INTEGRATOR_H__

#include "../../geometry/element.h"
#include "../../geometry/triangulation.h"
#include "../../geometry/interval.h"
#include "../../fields/scalar_expressions.h"
#include "../compile_time.h"
#include "../symbols.h"
#include "integrator_tables.h"
#include "../../pde/symbols.h"

namespace fdapde {
namespace core {

// A set of utilities to perform numerical integration
// T: integrator family tag, M: dimension of the integration space, R: order of basis elements
template <typename T, int LocalDim, int Order> class Integrator;

// specialization for Finite Element integration
template <int LocalDim, int Order> class Integrator<FEM, LocalDim, Order> {
   private:
    static constexpr int K = standard_fem_quadrature_rule<LocalDim, Order>::K;   // number of quadrature nodes
    IntegratorTable<LocalDim, K> integration_table_;
   public:
    Integrator() : integration_table_(IntegratorTable<LocalDim, K>()) {};

    // integrate a callable F over a mesh element e
    template <typename MeshType, typename ExprType>
    double integrate(const Element<MeshType>& e, const ExprType& f) const {
        double value = 0;
        for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
            if constexpr (std::is_invocable_r<double, ExprType, SVector<MeshType::embed_dim>>::value) {   // callable
                // map quadrature point onto e
                SVector<MeshType::embed_dim> p = e.J() * integration_table_.nodes[iq] + e.node(0);
                value += f(p) * integration_table_.weights[iq];
            } else {
                // as a fallback we assume f given as vector with the assumption that
                // f[integration_table_.num_nodes*e.id() + iq] equals the discretized field at the iq-th quadrature node
                value += f(integration_table_.num_nodes * e.id() + iq, 0) * integration_table_.weights[iq];
            }
        }
        // correct for measure of domain (element e)
        return value * e.measure();
    }
    // integrate a callable F over a mesh m
    template <typename MeshType, typename ExprType> double integrate(const MeshType& m, const ExprType& f) const {
        double value = 0;
        // cycle over all mesh elements
        for (const auto& e : m) value += integrate(e, f);
        return value;
    }
    // perform integration of \int_e [f * \phi] using a basis system defined over the reference element and the change
    // of variables formula: \int_e [f(x) * \phi(x)] = \int_{E} [f(J(X)) * \Phi(X)] |detJ| where J is the affine mapping
    // from the reference element E to the physical element e
    template <typename MeshType, typename ExprType, typename BasisType>
    double integrate(const Element<MeshType>& e, const ExprType& f, const BasisType& Phi) const {
        double value = 0;
        for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
            const SVector<MeshType::local_dim>& p = integration_table_.nodes[iq];
            if constexpr (std::is_base_of<ScalarExpr<MeshType::embed_dim, ExprType>, ExprType>::value) {
                // functor f is evaluable at any point.
                SVector<MeshType::embed_dim> Jp = e.J() * p + e.node(0);   // map quadrature point on physical element
                value += (f(Jp) * Phi(p)) * integration_table_.weights[iq];
            } else {
                // as a fallback we assume f given as vector of values with the assumption that
                // f[integration_table_.num_nodes*e.id() + iq] equals the discretized field at the iq-th quadrature node
                value += (f(integration_table_.num_nodes * e.id() + iq, 0) * Phi(p)) * integration_table_.weights[iq];
            }
        }
        // correct for measure of domain (element e)
        return value * e.measure();
    }
    // integrate the weak form of operator L to produce its (i,j)-th discretization matrix element
    template <typename L, typename MeshType, typename ExprType>
    double integrate(const Element<MeshType>& e, ExprType& f) const {
        // apply quadrature rule
        double value = 0;
        for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
            const SVector<MeshType::local_dim>& p = integration_table_.nodes[iq];
            if constexpr (std::remove_reference<L>::type::is_space_varying) {
                // space-varying case: forward the quadrature node index to non constant coefficients
                f.forward(integration_table_.num_nodes * e.id() + iq);
            }
            value += f(p) * integration_table_.weights[iq];
        }
        // correct for measure of domain (element e)
        return value * e.measure();
    }

    // getters
    template <typename MeshType> DMatrix<double> quadrature_nodes(const MeshType& m) const {
        DMatrix<double> quadrature_nodes;
        quadrature_nodes.resize(m.n_elements() * integration_table_.num_nodes, MeshType::embed_dim);
        // cycle over all mesh elements
        for (const auto& e : m) {
            // for each quadrature node, map it onto the physical element e and store it
            for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
                quadrature_nodes.row(integration_table_.num_nodes * e.id() + iq) =
                  e.J() * SVector<LocalDim>(integration_table_.nodes[iq].data()) + e.node(0);
            }
        }
        return quadrature_nodes;
    }
    std::size_t num_nodes() const { return integration_table_.num_nodes; }
};

// specialization for 1D spline integration (R: order of spline)
template <int Order> class Integrator<SPLINE, 1, Order> {
   private:
    static constexpr int K = 3;   // number of quadrature nodes (TODO: generalize)
    IntegratorTable<1, K, GaussLegendre> integration_table_;
   public:
    // integration of f over 1D segments of type [a,b], using formula
    // \int_{[a,b]} f(x) -> (b-a)/2 * \sum_{iq} w_{iq} * f((b-a)/2*x + (b+a)/2)
    template <typename ExprType> double integrate(double a, double b, const ExprType& f) const {
        double value = 0;
        for (std::size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
            value += f(SVector<1>(((b - a) / 2) * integration_table_.nodes[iq][0] + (b + a) / 2)) *
                     integration_table_.weights[iq];
        }
        // correct for measure of interval
        return (b - a) / 2 * value;
    }
    template <typename ExprType> double integrate(const Element<Triangulation<1, 1>>& e, const ExprType& f) const {
        return integrate(e.node(0), e.node(1), f);
    }
    // integrate a callable F over a 1D Mesh
    template <typename ExprType> double integrate(const Triangulation<1, 1>& m, const ExprType& f) const {
        double value = 0;
        // cycle over all mesh elements
        for (Triangulation<1, 1>::cell_iterator it = m.cells_begin(); it != m.cells_end(); ++it) {
            value += integrate(*it, f);
	}
        return value;
    }
    // getters
    DMatrix<double> quadrature_nodes(const Triangulation<1, 1>& m) const {
        DMatrix<double> quadrature_nodes;
        quadrature_nodes.resize(m.n_cells() * integration_table_.num_nodes, 1);
        // cycle over all mesh elements
        for (Triangulation<1, 1>::cell_iterator it = m.cells_begin(); it != m.cells_end(); ++it) {
            // for each quadrature node, map it onto the physical element e and store it
            for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
                quadrature_nodes.row(integration_table_.num_nodes * it->id() + iq) =
                  ((it->node(1) - it->node(0)) / 2) * integration_table_.nodes[iq][0] +
                  ((it->node(1) + it->node(0)) / 2);
            }
        }
        return quadrature_nodes;
    }
    std::size_t num_nodes() const { return integration_table_.num_nodes; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __INTEGRATOR_H__
