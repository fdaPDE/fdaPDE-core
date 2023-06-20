#ifndef __BARYCENTRIC_WALK_H__
#define __BARYCENTRIC_WALK_H__

#include "../../Mesh.h"
#include <random>

namespace fdaPDE{
namespace core{
namespace MESH{

  // implementation of a barycentric walk strategy for mesh elements. Does not work for manifold meshes.
  template <unsigned int M, unsigned int N, unsigned int R>
  class BarycentricWalk{
  private:
    const Mesh<M,N,R>& mesh_; 
  public:
    BarycentricWalk(const Mesh<M,N,R>& mesh) : mesh_(mesh) {};
    // applies a barycentric walk strategy to search for the element containing a given point
    template <typename... Args>
    std::shared_ptr<Element<M,N,R>> search(const SVector<N>& point, Args&... args) const;
  };

#include "BarycentricWalk.tpp"
}}}
  
#endif // __BARYCENTRIC_WALK_H__
