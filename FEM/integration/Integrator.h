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
template <unsigned int M, unsigned int K = standard_quadrature_rule<M>::K>
class Integrator {
 private:
  // reference to node and weights of the quadrature rule
  IntegratorTable<M, K> integrationTable_;
 public:
  // constructor
  Integrator() : integrationTable_(IntegratorTable<M, K>()) {};
  // integrate a callable F over a mesh element e
  template <unsigned int N, unsigned int R, typename F>
  double integrate(const Element<M, N, R>& e, F& f) const;
  // integrate a callable F over the entire mesh m
  template <unsigned int N, unsigned int R, typename F>
  double integrate(const Mesh<M, N, R>& m, const F& f) const;
  // integrate a BilinearFormExpr to produce the (i,j)-th element of its discretization
  template <unsigned int N, unsigned int R, typename B, typename F>
  double integrate(const B& basis, const Element<M, N, R>& e, int i , int j, const F& bilinearForm) const;
  // returns the integration table data structure
  const IntegratorTable<M, K>& getTable() const { return integrationTable_; }
};

#include "Integrator.tpp"

#endif // __INTEGRATOR_H__
