#ifndef __EVALUATOR_H__
#define __EVALUATOR_H__

#include <cstddef>
#include <vector>
#include "../utils/Symbols.h"
#include "PDE.h"
using fdaPDE::core::FEM::PDE;
#include "../MESH/engines/AlternatingDigitalTree/ADT.h"
using fdaPDE::core::MESH::ADT;
#include "../MESH/Element.h"
using fdaPDE::core::MESH::Element;
using fdaPDE::core::MESH::ct_nnodes;

namespace fdaPDE {
namespace core{
namespace FEM{

  // a raster is a collection of pixelized information where each pixel maps the value of a callable object in a given region of space.
  // raster data storage assume that pixels are referred to N-dimensional points arranged in an uniform grid placed at distance h_.
  template <unsigned int N>
  struct Raster{
    // the actual data to store
    std::vector<double> data_{};
    // points where the solution is evaluated.
    std::vector<SVector<N>> coords_{};
    double h_{}; // resolution used to produce the image

    // constructor
    Raster() = default;
    Raster(const std::vector<double>& data, const std::vector<SVector<N>>& coords, double h)
      : data_(data), coords_(coords), h_(h) {};
  };
  
  // a set of utilities to evaluate a PDE
  template <unsigned int M, unsigned int N, unsigned int R, typename SearchEngine = ADT<M,N>>
  class Evaluator{
  public:
    // constructor
    Evaluator() = default;

    // produce a raster image of the solution over the PDE domain using the specified resolution h
    template <typename E>
    Raster<N> toRaster(const PDE<M, N, R, E>& pde, double h) const;
    
  };

  template <unsigned int M, unsigned int N, unsigned int R, typename SearchEngine>
  template <typename E>
  Raster<N> Evaluator<M,N,R,SearchEngine>::toRaster(const PDE<M, N, R, E>& pde, double h) const {
    DVector<double> solution = pde.solution();
    // build search engine
    SearchEngine engine(pde.domain());
    // define mesh domain limits
    std::array<std::pair<double, double>, N> domain = pde.domain().range();
    // cycle until requested size is not reached
    std::vector<unsigned> currentSize(N, 0);
    SVector<N> p{}; // evaluation point and increment vector
    double totalSize = 1; // total number of pixels
    for(std::size_t i = 0; i < N; ++i){
      p[i] = domain[i].first;
      totalSize *= std::ceil((domain[i].second - domain[i].first)/h) + 1;
    }
    // preallocate memory to avoid useless reallocations
    std::vector<double> data{};
    std::vector<SVector<N>> coords{};
    data.resize(totalSize);
    coords.resize(totalSize);
    
    // start evaluation
    for(std::size_t i = 0; i < totalSize; ++i){
      // search element containing point
      std::shared_ptr<Element<M,N,R>> e = engine.search(p);
      double v = 0;    
      if(e != nullptr){
	// build a lagrangian basis over the element
	LagrangianBasis<2,2,1> basis(*e);
	// evaluate the solution at point
	for(size_t j = 0; j < ct_nnodes(M,R); ++j){
	  v += solution[e->nodeIDs()[j]] * basis[j](p);
	}
      }
      // store result
      data[i] = v;
      coords[i] = p;
      // update evaluation point
      for(std::size_t j = 0; j < N; ++j){
	p[j] += h;
	if(p[j] > domain[j].second){
	  p[j] = domain[j].first;
	}else break;
      }
    }
    // return raster image
    return Raster<N>(data, coords, h);
  }
  
}}}
#endif // __EVALUATOR_H__
