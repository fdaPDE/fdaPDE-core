#ifndef __QUERY_H__
#define __QUERY_H__

namespace fdaPDE{
namespace core{
namespace MESH{

  template <unsigned int N> using rectangle = std::pair<SVector<N>, SVector<N>>;

  // in geometric search a query is an N-dimensional rectangle described by its left-lower corner
  // and upper-right corner. This class wraps a rectangle providing some facilities to perform geometric
  // operations between queries
  template <unsigned int N> class Query {
  private:
    rectangle<N> queryRange_;
  public:
    // constructor
    Query(const rectangle<N>& queryRange) : queryRange_(queryRange) {}

    // returns true if a given point lies inside the query
    bool contains(const SVector<N>& point) const;
    // returns true if the query intersects a given rectangle
    bool intersect(const rectangle<N>& rectangle) const; 

    // getter
    rectangle<N> getRange() const { return queryRange_; }
  };

#include "Query.tpp"
}}}
  
#endif // __QUERY_H__
