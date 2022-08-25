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
  template <unsigned int M, unsigned int N>
  class BruteForce{
  private:
    Mesh<M,N>& mesh_;
  public:
    BruteForce(Mesh<M,N>& mesh) : mesh_(mesh) {} ;
    // applies a brute force strategy to search for the element containing a given point
    template <typename... Args>
    std::unique_ptr<Element<M, N>> search(const SVector<N>& point, Args&... args);
  };

#include "BruteForce.tpp"
}}}
  
#endif // __BRUTEFORCE_H__
