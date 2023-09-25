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

namespace fdapde {
namespace core {

// A set of utilities to perform numerical integration
// M: dimension of the domain of integration, R finite element order, K number of quadrature nodes
template <int M, int R, int K = standard_fem_quadrature_rule<M, R>::K> class Integrator {
   private:
    IntegratorTable<M, K> integration_table_;
   public:
    Integrator() : integration_table_(IntegratorTable<M, K>()) {};

    // integrate a callable F over a mesh element e
    template <int N, typename F> double integrate(const Element<M, N, R>& e, const F& f) const;
    // integrate a callable F over a triangualtion m
    template <int N, typename F> double integrate(const Mesh<M, N, R>& m, const F& f) const;
    // computes \int_e [f * \phi] where \phi is a basis function over the *reference element*.
    template <int N, typename F, typename B>
    double integrate(const Element<M, N, R>& e, const F& f, const B& phi) const;
    // integrate the weak form of operator L to produce its (i,j)-th discretization matrix element
    template <typename L, int N, typename F> double integrate(const Element<M, N, R>& e, F& f) const;

    // getters
    template <int N> DMatrix<double> quadrature_nodes(const Mesh<M, N, R>& m) const;
    std::size_t num_nodes() const { return integration_table_.num_nodes; }
};

// implementative details

// integration of bilinear form
template <int M, int R, int K>
template <typename L, int N, typename F>
double Integrator<M, R, K>::integrate(const Element<M, N, R>& e, F& f) const {
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

// perform integration of \int_e [f * \phi] using a basis system defined over the reference element and the change of
// variables formula: \int_e [f(x) * \phi(x)] = \int_{E} [f(J(X)) * \Phi(X)] |detJ|
// where J is the affine mapping from the reference element E to the physical element e
template <int M, int R, int K>
template <int N, typename F, typename B>
double Integrator<M, R, K>::integrate(const Element<M, N, R>& e, const F& f, const B& Phi) const {
    double value = 0;
    for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
        const SVector<M>& p = integration_table_.nodes[iq];
        if constexpr (std::is_base_of<ScalarExpr<N, F>, F>::value) {
            // functor f is evaluable at any point.
            SVector<N> Jp = e.barycentric_matrix() * p + e.coords()[0];   // map quadrature point on physical element e
            value += (f(Jp) * Phi(p)) * integration_table_.weights[iq];
        } else {
            // as a fallback we assume f given as vector of values with the assumption that
            // f[integration_table_.num_nodes*e.ID() + iq] equals the discretized field at the iq-th quadrature node.
            value += (f(integration_table_.num_nodes * e.ID() + iq, 0) * Phi(p)) * integration_table_.weights[iq];
        }
    }
    // correct for measure of domain (element e)
    return value * e.measure();
}

// integrate a callable F over a mesh element e. Do not require any particular structure for F
template <int M, int R, int K>
template <int N, typename F>
double Integrator<M, R, K>::integrate(const Element<M, N, R>& e, const F& f) const {
    double value = 0;
    for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
        if constexpr (std::is_invocable<F, SVector<N>>::value) {
            // functor f is evaluable at any point
            SVector<N> p =
              e.barycentric_matrix() * integration_table_.nodes[iq] + e.coords()[0];   // map quadrature point onto e
            value += f(p) * integration_table_.weights[iq];
        } else {
            // as a fallback we assume f given as vector of values with the assumption that
            // f[integration_table_.num_nodes*e.ID() + iq] equals the discretized field at the iq-th quadrature node.
            value += f(integration_table_.num_nodes * e.ID() + iq, 0) * integration_table_.weights[iq];
        }
    }
    // correct for measure of domain (element e)
    return value * e.measure();
}

// integrate a callable F over the entire mesh m.
template <int M, int R, int K>
template <int N, typename F>
double Integrator<M, R, K>::integrate(const Mesh<M, N, R>& m, const F& f) const {
    double value = 0;
    // cycle over all mesh elements
    for (const auto& e : m) value += integrate(e, f);
    return value;
}

// returns all quadrature points on the mesh
template <int M, int R, int K>
template <int N>
DMatrix<double> Integrator<M, R, K>::quadrature_nodes(const Mesh<M, N, R>& m) const {
    DMatrix<double> quadrature_nodes;
    quadrature_nodes.resize(m.elements() * integration_table_.num_nodes, N);
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

// integration of f() over 1D segments of type [a,b], using formula
// \int_{[a,b]} f(x) -> (b-a)/2 * \sum_{iq} w_{iq} * f((b-a)/2*x + (b+a)/2)
// and quadrature rule T
template <typename F, typename T> double integrate_1D(double a, double b, const F& f, const T& t) {
    static_assert(T::input_dim == 1, "quadrature rule input_dim != 1");
    double value = 0;
    for (std::size_t iq = 0; iq < t.num_nodes; ++iq) {
        value += f(SVector<1>(((b - a) / 2) * t.nodes[iq][0] + (b + a) / 2)) * t.weights[iq];
    }
    // correct for measure of interval
    return (b - a) / 2 * value;
}

}   // namespace core
}   // namespace fdapde

#endif   // __INTEGRATOR_H__
