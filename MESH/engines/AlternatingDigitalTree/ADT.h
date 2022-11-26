#ifndef __ADT_H__
#define __ADT_H__

#include <list>
#include <stack>
#include <set>

#include "../../../utils/Symbols.h"
#include "../../../utils/DataStructures/Tree.h"
using fdaPDE::core::Tree;
#include "../../Mesh.h"
#include "Query.h"

namespace fdaPDE{
namespace core{
namespace MESH{

  // a specific node data structure for easy management of the ADT during element search
  template <unsigned int N>
  struct ADTnode {
    unsigned int elementID_; // the element ID to which this node referes to
    SVector<N> point_; // the point stored in this node 
    rectangle<N> range_; // the range in the unit hypercube this node refers to
    
    ADTnode(unsigned int elementID, const SVector<N>& point, const rectangle<N>& range)
      : elementID_(elementID), point_(point), range_(range) {}
  };

  // Alternating Digital Tree implementation for tree-based search of elements over an unstructured mesh
  template <unsigned int M, unsigned int N, unsigned int R>
  class ADT{
  private:
    Tree<ADTnode<2*N>> tree; // tree data structure to support ADT
    const Mesh<M,N,R>& mesh_; // domain over which build the ADT
    std::array<double, N> normalization_; // vector of range-normalization constants

    // build an Alternating Digital Tree given a set of 2N-dimensional points.
    void init(const std::vector<std::pair<SVector<2*N>, unsigned int>>& data);
    // performs a geometric search returning all points which lie in a given query
    std::list<unsigned int> geometricSearch(const Query<2*N>& query) const;
  public:
    ADT(const Mesh<M,N,R>& mesh);  
    // applies the ADT geometric search to return the mesh element containing a given point
    template <typename... Args>
    std::shared_ptr<Element<M,N,R>> search(const SVector<N>& point, Args&... args) const;

    // getter
    Tree<ADTnode<2*N>> getTree() const { return tree; }
  };

#include "ADT.tpp"
}}}

#endif // __ADT_H__
