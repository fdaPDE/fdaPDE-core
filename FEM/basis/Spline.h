#ifndef __SPLINE_H__
#define __SPLINE_H__

#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <queue>

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

// a node of a spline tree (an N_ij object in the above schema)
class SplineNode {
private:
  double k1_, k2_; // knots u_i   and u_i+j
  double d1_ = 0;  // for efficiency reasons precompute 1/(u_i+j - u_i)

public:
  // constructor
  SplineNode() = default;
  SplineNode(double k1, double k2) : k1_(k1), k2_(k2), d1_(1/(k2_ - k1_)){ }

  // for a spline of order j get the knot span [u_i, u_i+j)
  std::pair<double, double> getKnotSpan() const {
    return std::make_pair(k1_, k2_);
  }
  
  double operator()(double x) const { return (x - k1_)*d1_; }

  // returns true if the point x belongs to the knot span of N_ij
  bool contains(double x) const {
    if(k1_ < k2_) return (x >= k1_ && x < k2_) ? true : false;
    else          return (x >= k2_ && x < k1_) ? true : false;
  }
};

// functor passed to the DFS visit. See Tree.h for more details
struct SplineEvaluator{
  double result_ = 0;
  std::map<int, double> partialMap_{};
  double partialProduct_ = 1;
  double inputPoint_;
    
  SplineEvaluator(const node_ptr<SplineNode>& root_ptr, double inputPoint) : inputPoint_(inputPoint) {
    partialMap_[root_ptr->getKey()] = 1;
  };

  // add result only if N_i0(x) = 1, that is if x is contained in the knot span [u_i, u_i+1)
  void leafAction(const node_ptr<SplineNode>& currentNode) {
    if(currentNode->getData().contains(inputPoint_))
      result_ += partialProduct_;

    // restore partialProduct to the one of the father
    partialProduct_ = partialMap_[currentNode->getFather()->getKey()];
    return;
  }

  /* A spline is defined by the recursive formula

     [(x-u_i)/(u_i+j - u_i)]*N_i,j-1(x) or [(u_i+j+1 - x)/(u_i+j+1 - u_i+1)]*N_i+1,j-1(x)

     At the end of recursion the spline equation can be seens as N_ij = w0*N_00 + w1*N_10 + ... + wM*N_M0
     
     During DFS visit coefficients w0, w1, ..., wM are computed from root to leafs. firstVisitAction() just develops the coefficients
     product of each term N_i0.
  */
  void firstVisitAction(const node_ptr<SplineNode>& currentNode) {
    // update partial product
    partialProduct_ *= currentNode->getData()(inputPoint_);
    partialMap_[currentNode->getKey()] = partialProduct_;
    return;
  }

  // go back in the recursion tree
  void lastVisitAction(const node_ptr<SplineNode>& currentNode) {
    // restore to partialProduct of the father
    partialProduct_ = partialMap_[currentNode->getFather()->getKey()];
    return;
  }
};
  
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

  // build a spline tree given the knot vector and a pair of indexes i,j,
  // where i denotes the knot the spline refers to and j the spline order
  Spline(const std::vector<double>& knotVector, int i, int j);

  // initialize a spline from a vector of spline trees and associated weights (used by gradient() method)
  Spline(const std::vector<double>& knotVector, const std::vector<std::tuple<double, Tree<SplineNode>, int, int> >& spline)
    : knotVector_(knotVector), spline_(spline) {}
  
  double operator()(double) const;    // evaluates the spline vector at a given point
  Spline gradient() const;            // compute derivative of spline. This is another spline object
};

#include "Spline.tpp"

#endif // __SPLINE_H__
