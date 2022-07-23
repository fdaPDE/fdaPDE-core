#ifndef __SEARCH_ACTIONS_H__
#define __SEARCH_ACTIONS_H__

#include <unordered_map>
#include <memory>
#include "../../utils/Symbols.h"
#include "../Element.h"
using fdaPDE::core::MESH::Element;

// This file contains a collection of usefull functors to assist the searching process.
// Let e the element containing the queried point p, each search engine accepts one or more functors with the following signature
//         void operator()(const std::shared_ptr<Element<M,N>>& e, const SVector<N>& p);
// allowing to perform custom actions at the end of the search process on the pair (e,p)

namespace fdaPDE {
namespace core{
namespace MESH{

  // saves ID of the element containing p and the barycentric coordinates of p with respect to e
  template <unsigned int M, unsigned int N>
  struct StandardQueryRecorder {
    std::unordered_map<std::size_t, SVector<M+1>> result{};
    // constructor
    StandardQueryRecorder() = default;

    void operator()(const std::shared_ptr<Element<M,N>>& e, const SVector<N>& p){
      result[e->ID()] = e->toBarycentricCoords(p);
    }
  };
 
}}}

#endif // __SEARCH_ACTIONS_H__
