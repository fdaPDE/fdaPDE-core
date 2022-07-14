#ifndef __LAGRANGIAN_BASIS_H__
#define __LAGRANGIAN_BASIS_H__

#include "../utils/CompileTime.h"
#include "../utils/Symbols.h"
#include "MultivariatePolynomial.h"
#include <array>
#include <Eigen/QR>

// hard code the coordinates of nodes over an N-dimensional unit simplex for Lagrange interpolation
// N dimension of the simplex, R order of the basis to be defined on it
template <unsigned int N, unsigned int R> struct ReferenceNodes;
template <unsigned int N, unsigned int R> using point_list = std::array<std::array<double, N>, R>;

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
template <unsigned int N, unsigned int R>
class LagrangianBasis {
private:
  // nodes of the lagrangian basis
  std::array<std::array<double, N>, ct_binomial_coefficient(R + N, R)> nodes_;
  // a Lagrangian basis is just a collection of properly defined polynomials
  std::array<MultivariatePolynomial<N,R>, ct_binomial_coefficient(N+R,R)> basis_;

  void computeCoefficients(const std::array<std::array<double, N>, ct_binomial_coefficient(N+R, R)>& nodes);
  
public:
  // expose basis order
  static constexpr unsigned int order_ = R;
  
  // a Lagrangian basis built over a given set of nodes
  LagrangianBasis(const std::array<std::array<double, N>, ct_binomial_coefficient(N+R, R)>& nodes) : nodes_(nodes) {
    computeCoefficients(nodes_);
  };

  // default constructor constructs a Lagrangian basis built over the referece N-dimensional unit simplex
  LagrangianBasis() : LagrangianBasis<N, R>(ReferenceNodes<N, R>::nodes) {};

  // construct a Lagrangian basis over a mesh element e
  template <unsigned int M, unsigned int K>
  LagrangianBasis(const Element<M, K>& e){
    // collect nodes from mesh element
    std::array<std::array<double, N>, ct_binomial_coefficient(N+R, R)> elementNodes{};
    for(size_t j = 0; j < e.getFESupport().size(); ++j)
      elementNodes[j] = {e.getFESupport()[j].second[0], e.getFESupport()[j].second[1]};

    // call computing coefficients routine
    computeCoefficients(elementNodes);    
    return;
  }

  // subscript operator to directly access basis elements
  const MultivariatePolynomial<N, R>& operator[](size_t i) const { return basis_[i]; }
  int size() const { return basis_.size(); }  // return the number of basis elements
  
};

template <unsigned int N, unsigned int R>
void LagrangianBasis<N, R>::computeCoefficients(const std::array<std::array<double, N>, ct_binomial_coefficient(N+R, R)>& nodes){
    // build vandermonde matrix
    constexpr unsigned int M = ct_binomial_coefficient(N+R,R);
    constexpr std::array<std::array<unsigned, N>, M> expTable_ = MultivariatePolynomial<N,R>::expTable_;

    // Vandermonde matrix construction
    SMatrix<M> V = Eigen::Matrix<double, M, M>::Ones();
    for(size_t i = 0; i < M; ++i){
      for(size_t j = 1; j < M; ++j){
	V(i,j) = MonomialProduct<N-1, std::array<double, N>, std::array<unsigned, N>>::unfold(nodes_[i], expTable_[j]);
      }
    }
    
    // solve system V*a = b with b vector having 1 at position i and 0 everywhere else.
    // Its solution gives the vector of coefficients of the i-th Lagrange polynomial
    Eigen::ColPivHouseholderQR<SMatrix<M>> QRdecomposition(V);
    for(size_t i = 0; i < M; ++i){
      // build rhs vector
      SVector<M> b = Eigen::Matrix<double, M, 1>::Zero();
      b[i] = 1;

      SVector<M> coeff = QRdecomposition.solve(b); // solve system

      // cast to array
      std::array<double, M> coeff_array;
      for(size_t j = 0; j < M; ++j) coeff_array[j] = coeff[j];
      
      basis_[i] = MultivariatePolynomial<N, R>(coeff_array); // store basis
    }
    return;
}


#endif // __LAGRANGIAN_BASIS_H__
