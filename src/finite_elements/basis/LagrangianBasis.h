#ifndef __LAGRANGIAN_BASIS_H__
#define __LAGRANGIAN_BASIS_H__

#include "../../utils/CompileTime.h"
#include "../../utils/Symbols.h"
#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;
using fdaPDE::core::MESH::ct_nnodes;
#include "../../MESH/ReferenceElement.h"
using fdaPDE::core::MESH::ReferenceElement;
#include "MultivariatePolynomial.h"
using fdaPDE::core::FEM::MultivariatePolynomial;

namespace fdaPDE{
namespace core{
namespace FEM{
    
  // A class representing a Lagrangian Basis of degree R over an M-dimensional space defined over a given set of N-dimensional nodes.
  // It uses the Vandermonde matrix to compute coefficients of lagrange polynomials
  template <unsigned int M, unsigned int N, unsigned int R>
  class LagrangianBasis {
  private:
    // nodes of the lagrangian basis (std::array is constexpr evaluable, use std::array instead of SVector<N>)
    std::array<std::array<double, M>, ct_binomial_coefficient(M+R,R)> nodes_;
    // for a basis defined over a mesh element, nodeIDs_[i] is the mesh node where the i-th element of the basis is 1
    std::array<std::size_t, ct_binomial_coefficient(M+R,R)> nodeIDs_;
    std::array<MultivariatePolynomial<M,R>, ct_binomial_coefficient(M+R,R)> basis_; // the actual basis
    // computes coefficients of polynomials composing the basis by solution of linear system
    void computeCoefficients(const std::array<std::array<double, M>, ct_binomial_coefficient(M+R,R)>& nodes);
  
  public:
    // expose basis order
    static constexpr unsigned int order = R;
    typedef MultivariatePolynomial<M,R> element_type;
    using const_iterator = typename std::array<MultivariatePolynomial<M,R>, ct_binomial_coefficient(M+R,R)>::const_iterator;
    
    // a Lagrangian basis built over a given set of nodes
    LagrangianBasis(const std::array<std::array<double, M>, ct_binomial_coefficient(M+R,R)>& nodes) : nodes_(nodes) {
      computeCoefficients(nodes_);
    };
    // a Lagrangian basis built over the referece N-dimensional unit simplex
    LagrangianBasis() : LagrangianBasis<M,N,R>(ReferenceElement<M,R>::nodes) {};
  
    // subscript operator to directly access basis elements
    const MultivariatePolynomial<M,R>& operator[](size_t i) const { return basis_[i]; }
    // return the number of basis elements
    int size() const { return basis_.size(); }
    // allow range-for over basis elements
    const_iterator begin() const;
    const_iterator end() const;
    // return mesh nodes where basis is defined
    std::array<std::size_t, ct_binomial_coefficient(M+R,R)> nodes() const { return nodeIDs_; }
  };
  
  #include "LagrangianBasis.tpp"
}}}

#endif // __LAGRANGIAN_BASIS_H__
