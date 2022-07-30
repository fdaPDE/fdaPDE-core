#ifndef __LAGRANGIAN_BASIS_H__
#define __LAGRANGIAN_BASIS_H__

#include "../../utils/CompileTime.h"
#include "../../utils/Symbols.h"
#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;
using fdaPDE::core::MESH::ct_nnodes;
#include "MultivariatePolynomial.h"
#include <array>
#include <Eigen/QR>

namespace fdaPDE{
namespace core{
namespace FEM{
  
  // hard code the coordinates of nodes over an N-dimensional unit simplex for Lagrange interpolation for linear and
  // quadratic finite elements
  // M dimension of the simplex, R order of the basis to be defined on it
  template <unsigned int M, unsigned int R> struct ReferenceNodes;
  template <unsigned int M, unsigned int R> using point_list = std::array<std::array<double, M>, R>;

  template<> // 1D first order basis
  struct ReferenceNodes<1,1>{
    static constexpr point_list<1,2> nodes  = {
      {{0}, {1}}
    };};
  template<> // 1D second order basis
  struct ReferenceNodes<1,2>{
    static constexpr point_list<1,3> nodes  = {
      {{0}, {0.5}, {1}}
    };};

  template<> // 2D first order basis
  struct ReferenceNodes<2,1>{
    static constexpr point_list<2,3> nodes  = {
      {{0, 0}, {1, 0}, {0, 1}}
    };};
  template<> // 2D second order basis
  struct ReferenceNodes<2,2>{
    static constexpr point_list<2,6> nodes  = {
      {{0, 0}, {1, 0}, {0, 1}, {0.5, 0.5}, {0, 0.5}, {0.5, 0}}
    };};

  template<> // 3D first order basis
  struct ReferenceNodes<3,1>{
    static constexpr point_list<3,4> nodes  = {
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
    };};
  template<> // 3D second order basis
  struct ReferenceNodes<3,2>{
    static constexpr point_list<3,10> nodes = {
      {{0, 0, 0},   {1, 0, 0},   {0, 1, 0},     {0, 0, 1},     {0.5, 0.5, 0},
       {0, 0.5, 0}, {0.5, 0, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}, {0, 0, 0.5}}
    };};
  
  // A class representing a Lagrangian Basis defined over a given set of nodes.
  // It uses the Vandermonde matrix to compute coefficients of lagrange polynomials
  template <unsigned int M, unsigned int R>
  class LagrangianBasis {
  private:
    // nodes of the lagrangian basis (std::array is constexpr evaluable, use std::array instead of SVector<N>)
    std::array<std::array<double, M>, ct_binomial_coefficient(M+R,R)> nodes_;
    // a Lagrangian basis is just a collection of properly defined polynomials
    std::array<MultivariatePolynomial<M,R>, ct_binomial_coefficient(M+R,R)> basis_;
    // computes coefficients of polynomials compising the basis by solution of linear system
    void computeCoefficients(const std::array<std::array<double, M>, ct_binomial_coefficient(M+R,R)>& nodes);
  
  public:
    // expose basis order
    static constexpr unsigned int order = R;
    using const_iterator = typename std::array<MultivariatePolynomial<M,R>, ct_binomial_coefficient(M+R,R)>::const_iterator;
    
    // a Lagrangian basis built over a given set of nodes
    LagrangianBasis(const std::array<std::array<double, M>, ct_binomial_coefficient(M+R,R)>& nodes) : nodes_(nodes) {
      computeCoefficients(nodes_);
    };
    // a Lagrangian basis built over the referece N-dimensional unit simplex
    LagrangianBasis() : LagrangianBasis<M, R>(ReferenceNodes<M, R>::nodes) {};
    // construct a Lagrangian basis over a mesh element e, N in this case represents the local dimension of the element
    template <unsigned int N>
    LagrangianBasis(const Element<M,N,R>& e);
  
    // subscript operator to directly access basis elements
    const MultivariatePolynomial<M, R>& operator[](size_t i) const { return basis_[i]; }
    // return the number of basis elements
    int size() const { return basis_.size(); }
    // allow range-for over basis elements
    const_iterator begin() const;
    const_iterator end() const;    
  };

  #include "LagrangianBasis.tpp"
}}}

#endif // __LAGRANGIAN_BASIS_H__
