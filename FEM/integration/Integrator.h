#ifndef __INTEGRATOR_H__
#define __INTEGRATOR_H__

#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;
#include "../../MESH/Mesh.h"
using fdaPDE::core::MESH::Mesh;

#include "../../utils/Symbols.h"
#include "../../utils/CompileTime.h"
#include "../../utils/fields/VectorField.h"
using fdaPDE::core::VectorField;

#include "IntegratorTables.h"

// An integrator class. Just integrates a given integrable function (i.e. ScalarField or MultivariatePolynomials) on a mesh element.
// Given a function f() its integral \int_K f(x)dx can be approximated using a quadrature rule. The general
// scheme of a qudrature formula for the approximation of integral \int_K f(x)dx is given by a finite sum
// \sum_{i=1}^N [f(x_i) * w_i] where x_i and w_i are properly choosen quadrature nodes and weights.
// N space dimension, K number of nodes in the quadrature rule
template <unsigned int N, unsigned int K = select_standard_quadrature_rule<N>::K>
class Integrator {
 private:
  // reference to node and weights of the quadrature rule
  IntegratorTable<N, K> integrationTable_;
 public:
  // constructor
  Integrator() : integrationTable_(IntegratorTable<N, K>()) {};
  // integrate a callable F over a mesh element e
  template <unsigned int M, unsigned int R, typename F>
  double integrate(const Element<M, N, R>& e, F& f) const;
  // integrate a callable F over the entire mesh m
  template <unsigned int M, unsigned int L, typename F>
  double integrate(const Mesh<M, L>& m, const F& f) const;
  // integrate a BilinearFormExpr to produce the (i,j)-th element of its discretization
  template <unsigned int ORDER, typename B, typename F>
  double integrate(const B& basis, const Element<ORDER, N>& e, int i , int j, const F& bilinearForm) const;
  // returns the integration table data structure
  const IntegratorTable<N, K>& getTable() const { return integrationTable_; }
};

// integrate a BilinearFormExpr over mesh element e using basis B. The pair (i,j) indicates the element position of the produced
// value in the matrix discretizing the form. This method is used as part of the assembly loop in the computation of the
// discretization matrix of the differential operator L
template <unsigned int N, unsigned int K>
template <unsigned int ORDER, typename B, typename F>
double Integrator<N, K>::integrate(const B& basis, const Element<ORDER, N>& e, int i , int j, const F& bilinearForm) const{
  // apply quadrature rule
  double value = 0;
  for(size_t iq = 0; iq < integrationTable_.num_nodes; ++iq){
    // for a BilinearFormExpr .integrate() is developed in the sum of the single integrals yelding to the discretization of the bilinear form
    SVector<N> p = SVector<N>(integrationTable_.nodes[iq].data());
    value += bilinearForm.integrate(basis, e, i, j, p) * integrationTable_.weights[iq];
  }
  // correct for measure of domain (element e)
  return value * e.measure();
}

// integrate a callable F over a mesh element e.
template <unsigned int N, unsigned int K>
template <unsigned int M, unsigned int R, typename F>
double Integrator<N, K>::integrate(const Element<M, N, R> &e, F &f) const {
  double value = 0;
  // execute quadrature rule
  for(size_t iq = 0; iq < integrationTable_.num_nodes; ++iq){
    // map quadrature point to current element e
    SVector<N> p = e.barycentricMatrix()*SVector<M>(integrationTable_.nodes[iq].data()) + e.coords()[0];
    value += f(p)*integrationTable_.weights[iq];
  }
  // correct for measure of domain (element e)
  return value * e.measure();
}

// integrate a callable F over the entire mesh m.
// Just exploit linearity of the integral operation to sum the result of integrating F over each single mesh element
template <unsigned int N, unsigned int K>
template <unsigned int M, unsigned int L, typename F>
double Integrator<N,K>::integrate(const Mesh<M, L> &m, const F &f) const {
  double value = 0;
  for(const auto& element : m)
    value += integrate(element, f);
  return value;
}

#endif // __INTEGRATOR_H__
