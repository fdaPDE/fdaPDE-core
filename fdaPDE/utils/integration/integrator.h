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

#include "../../mesh/element.h"
#include "../../mesh/mesh.h"
#include "../../fields/scalar_expressions.h"
#include "../compile_time.h"
#include "../symbols.h"
#include "integrator_tables.h"
#include "../../pde/symbols.h"

namespace fdapde {
namespace core {

// A set of utilities to perform numerical integration
// T: integrator family tag, M: dimension of the integration space, R: order of basis elements
template <typename T, int M, int R> class Integrator;

// specialization for Finite Element integration
template <int M, int R> class Integrator<FEM, M, R> {
   private:
    static constexpr int K = standard_fem_quadrature_rule<M, R>::K;   // number of quadrature nodes
    IntegratorTable<M, K> integration_table_;
   public:
    Integrator() : integration_table_(IntegratorTable<M, K>()) {};

    // integrate a callable F over a mesh element e
    template <int N, typename F> double integrate(const Element<M, N>& e, const F& f) const {
        double value = 0;
        for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
            if constexpr (std::is_invocable_r<double, F, SVector<N>>::value) {
                // functor f is evaluable at any point
                SVector<N> p = e.barycentric_matrix() * integration_table_.nodes[iq] +
                               e.coords()[0];   // map quadrature point onto e
                value += f(p) * integration_table_.weights[iq];
            } else {
                // as a fallback we assume f given as vector of values with the assumption that
                // f[integration_table_.num_nodes*e.ID() + iq] equals the discretized field at the iq-th quadrature
                // node.
                value += f(integration_table_.num_nodes * e.ID() + iq, 0) * integration_table_.weights[iq];
            }
        }
        // correct for measure of domain (element e)
        return value * e.measure();
    }
    // integrate a callable F over a triangualtion m
    template <int N, typename F> double integrate(const Mesh<M, N>& m, const F& f) const {
        double value = 0;
        // cycle over all mesh elements
        for (const auto& e : m) value += integrate(e, f);
        return value;
    }
    // perform integration of \int_e [f * \phi] using a basis system defined over the reference element and the change
    // of variables formula: \int_e [f(x) * \phi(x)] = \int_{E} [f(J(X)) * \Phi(X)] |detJ| where J is the affine mapping
    // from the reference element E to the physical element e
    template <int N, typename F, typename B> double integrate(const Element<M, N>& e, const F& f, const B& Phi) const {
        double value = 0;
        for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
            const SVector<M>& p = integration_table_.nodes[iq];
            if constexpr (std::is_base_of<ScalarExpr<N, F>, F>::value) {
                // functor f is evaluable at any point.
                SVector<N> Jp =
                  e.barycentric_matrix() * p + e.coords()[0];   // map quadrature point on physical element e
                value += (f(Jp) * Phi(p)) * integration_table_.weights[iq];
            } else {
                // as a fallback we assume f given as vector of values with the assumption that
                // f[integration_table_.num_nodes*e.ID() + iq] equals the discretized field at the iq-th quadrature
                // node.
                value += (f(integration_table_.num_nodes * e.ID() + iq, 0) * Phi(p)) * integration_table_.weights[iq];
            }
        }
        // correct for measure of domain (element e)
        return value * e.measure();
    }
    // integrate the weak form of operator L to produce its (i,j)-th discretization matrix element
    template <typename L, int N, typename F> double integrate(const Element<M, N>& e, F& f) const {
        // apply quadrature rule
        double value = 0;
        for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
            const SVector<M>& p = integration_table_.nodes[iq];
            if constexpr (std::remove_reference<L>::type::is_space_varying) {
                // space-varying case: forward the quadrature node index to non constant coefficients
                f.forward(integration_table_.num_nodes * e.ID() + iq);
            }
            value += f(p) * integration_table_.weights[iq];
        }
        // correct for measure of domain (element e)
        return value * e.measure();
    }

    // getters
    template <int N> DMatrix<double> quadrature_nodes(const Mesh<M, N>& m) const {
        DMatrix<double> quadrature_nodes;
        quadrature_nodes.resize(m.n_elements() * integration_table_.num_nodes, N);
        // cycle over all mesh elements
        for (const auto& e : m) {
            // for each quadrature node, map it onto the physical element e and store it
            for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
                quadrature_nodes.row(integration_table_.num_nodes * e.ID() + iq) =
                  e.barycentric_matrix() * SVector<M>(integration_table_.nodes[iq].data()) + e.coords()[0];
            }
        }
        return quadrature_nodes;
    }
    std::size_t num_nodes() const { return integration_table_.num_nodes; }
};

// specialization for 1D spline integration (R: order of spline)
template <int R> class Integrator<SPLINE, 1, R> {
   private:
    static constexpr int K = 3;   // number of quadrature nodes (TODO: generalize)
    IntegratorTable<1, K, GaussLegendre> integration_table_;
   public:
    // integration of f over 1D segments of type [a,b], using formula
    // \int_{[a,b]} f(x) -> (b-a)/2 * \sum_{iq} w_{iq} * f((b-a)/2*x + (b+a)/2)
    template <typename F> double integrate(double a, double b, const F& f) const {
        double value = 0;
        for (std::size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
            value += f(SVector<1>(((b - a) / 2) * integration_table_.nodes[iq][0] + (b + a) / 2)) *
                     integration_table_.weights[iq];
        }
        // correct for measure of interval
        return (b - a) / 2 * value;
    }
    template <typename F> double integrate(const Element<1, 1>& e, const F& f) const {
        return integrate(e.coords()[0], e.coords()[1], f);
    }
    // integrate a callable F over a 1D Mesh
    template <typename F> double integrate(const Mesh<1, 1>& m, const F& f) const {
        double value = 0;
        // cycle over all mesh elements
        for (const auto& e : m) value += integrate(e, f);
        return value;
    }
    // getters
    DMatrix<double> quadrature_nodes(const Mesh<1, 1>& m) const {
        DMatrix<double> quadrature_nodes;
        quadrature_nodes.resize(m.n_elements() * integration_table_.num_nodes, 1);
        // cycle over all mesh elements
        for (const auto& e : m) {
            // for each quadrature node, map it onto the physical element e and store it
            for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
                quadrature_nodes.row(integration_table_.num_nodes * e.ID() + iq) =
                  ((e.coords()[1] - e.coords()[0]) / 2) * integration_table_.nodes[iq][0] +
                  ((e.coords()[1] + e.coords()[0]) / 2);
            }
        }
        return quadrature_nodes;
    }
    std::size_t num_nodes() const { return integration_table_.num_nodes; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __INTEGRATOR_H__
