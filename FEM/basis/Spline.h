#ifndef __SPLINE_H__
#define __SPLINE_H__

#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <queue>

namespace fdaPDE{
namespace core{
namespace FEM{

/* Let u_0, u_1, ..., u_n n distinct knots. Call U = [u_0, u_1, ..., u_n] knot vector. Define B_ij the i-th spline basis of order j.
   By Cox-de Boor formula splines can be recursively defined as

   N_i0(x) = 1 if x \in [u_i, u_i+1) 0 otherwise
   N_ij(x) = [(x-u_i)/(u_i+j - u_i)]*N_i,j-1(x) + [(u_i+j+1 - x)/(u_i+j+1 - u_i+1)]*N_i+1,j-1(x)
             |---------------------|              |-------------------------------|
                       wL                                         wR
	     
   The above recursive formula can be efficiently stored using a binary tree structure. For example a cubic spline will be represented by the following
   recursion tree:

                          N_02
	   wL*N_01                    wR*N_11
   wL*N_00         wR*N_10    wL*N_10         wR*N_20
   
   Each node of the tree will contains the knots required for computing weights w1 and w2. Evaluation will take place using a
   DFS visit with a proper defined functor (see Tree.h for specifications on how the DFS works).
*/

  // forward declarations
  class SplineNode;
  struct SplineEvaluator;
  
  // just a collection of Tree<SplineNode>. Compliant with the functional basis concept adopted in the library.
  class Spline {
  private:
    // the actual spline. This is a vector since spline derivatives are stored as a a vector of lower degree spline trees
    std::vector<std::tuple<double, Tree<SplineNode>, int, int> > spline_{}; 
    int i_, j_;
    std::vector<double> knotVector_;

    // evaluate a single tree in the spline_ vector
    double eval(const Tree<SplineNode>& tree, double x) const;
    // extract i-th subtree
    Tree<SplineNode>& operator[](std::size_t i) { return std::get<1>(spline_[i]); } 
  public:
    // wraps an already existing spline tree in a spline object
    Spline(const Tree<SplineNode>& spline, int i, int j) { spline_.push_back(std::make_tuple(1, spline, i, j)); }
    // build a spline tree given the knot vector and a pair of indexes i,j, where i denotes the knot the spline refers to and j the spline order
    Spline(const std::vector<double>& knotVector, int i, int j);
    // initialize a spline from a vector of spline trees and associated weights (used by gradient() method)
    Spline(const std::vector<double>& knotVector, const std::vector<std::tuple<double, Tree<SplineNode>, int, int> >& spline)
      : knotVector_(knotVector), spline_(spline) {}
  
    double operator()(double) const; // evaluates the spline vector at a given point
    Spline gradient() const;         // compute derivative of spline as another spline object
  };

}}}
#endif // __SPLINE_H__
