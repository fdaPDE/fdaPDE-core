#ifndef __REFERENCE_ELEMENT_H__
#define __REFERENCE_ELEMENT_H__

#include <array>
#include "../utils/Symbols.h"

namespace fdaPDE {
namespace core {
namespace MESH {

  // Definition of the unit reference simplices in dimension M to support a functional basis of order R. Those are
  // extensively used in the internals of FEM to produce the discretization of a bilinear form
  template <unsigned int M, unsigned int R> struct ReferenceElement;
  template <unsigned int M, unsigned int R> using point_list = std::array<std::array<double, M>, R>;

  template<> // 1D first order basis
  struct ReferenceElement<1,1>{
    static constexpr point_list<1,2> nodes  = {
      {{0}, {1}}
    };};
  template<> // 1D second order basis
  struct ReferenceElement<1,2>{
    static constexpr point_list<1,3> nodes  = {
      {{0}, {0.5}, {1}}
    };};

  template<> // 2D first order basis
  struct ReferenceElement<2,1>{
    static constexpr point_list<2,3> nodes  = {
      {{0, 0}, {1, 0}, {0, 1}}
    };
    const std::array<SVector<3>,3> bary_coords = {
      SVector<3>(1,0,0), SVector<3>(0,1,0), SVector<3>(0,0,1)
    };

  };
  template<> // 2D second order basis
  struct ReferenceElement<2,2>{
    static constexpr point_list<2,6> nodes  = {
      {{0, 0}, {1, 0}, {0, 1}, {0, 0.5}, {0.5, 0}, {0.5, 0.5}}
    };
    const std::array<SVector<3>,6> bary_coords = {
      SVector<3>(1,0,0),     SVector<3>(0,1,0),     SVector<3>(0,0,1),
      SVector<3>(0.5,0.5,0), SVector<3>(0.5,0,0.5), SVector<3>(0,0.5,0.5)
    };
  };

  template<> // 3D first order basis
  struct ReferenceElement<3,1>{
    static constexpr point_list<3,4> nodes  = {
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
    };};
  template<> // 3D second order basis
  struct ReferenceElement<3,2>{
    static constexpr point_list<3,10> nodes = {
      {{0, 0, 0},   {1, 0, 0},   {0, 1, 0},     {0, 0, 1},     {0.5, 0.5, 0},
       {0, 0.5, 0}, {0.5, 0, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}, {0, 0, 0.5}}
    };};

  
}}}

#endif // __REFERENCE_ELEMENT_H__
