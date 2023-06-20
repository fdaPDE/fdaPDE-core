#ifndef __BRUTEFORCE_H__
#define __BRUTEFORCE_H__

#include "../../Mesh.h"
#include "../../Element.h"
#include <memory>

namespace fdaPDE{
namespace core{
namespace MESH{

  // bruteforce strategy for search elements over a mesh. This works under any assumption (requires just a .contains()
  // method provided by the mesh elements) anyway is much slower than other engines. Consider the use of an ADT based
  // engine instead.
  template <unsigned int M, unsigned int N, unsigned int R>
  class BruteForce{
  private:
    const Mesh<M,N,R>& mesh_;
  public:
    BruteForce(const Mesh<M,N,R>& mesh) : mesh_(mesh) {} ;
    // applies a brute force strategy to search for the element containing a given point
    template <typename... Args>
    std::shared_ptr<Element<M,N,R>> search(const SVector<N>& point, Args&... args) const;
  };

#include "BruteForce.tpp"
}}}
  
#endif // __BRUTEFORCE_H__
