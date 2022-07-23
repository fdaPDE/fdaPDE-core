#ifndef __BARYCENTRIC_WALK_H__
#define __BARYCENTRIC_WALK_H__

#include "../../Mesh.h"
#include <random>

namespace fdaPDE{
namespace core{
namespace MESH{

  // implementation of a barycentric walk strategy for mesh elements. Does not work for manifold meshes.
  template <unsigned int M, unsigned int N>
  class BarycentricWalk{
  private:
    const Mesh<M,N>& mesh_;
    // a random number generator is used as part of the initialization procedure of the walking algorithm
    uint32_t seed;
    std::default_random_engine rng;
    std::uniform_int_distribution<uint32_t> uniform_int;
 
  public:
    BarycentricWalk(const Mesh<M,N>& mesh);
    // applies a barycentric walk strategy to search for the element containing a given point
    template <typename... Args>
    std::shared_ptr<Element<M, N>> search(const SVector<N>& point, Args&... args);
  };

#include "BarycentricWalk.tpp"
}}}
  
#endif // __BARYCENTRIC_WALK_H__
