#ifndef __EVALUATOR_H__
#define __EVALUATOR_H__

#include <cstddef>
#include <type_traits>
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
    // return size of the image (number of points)
    std::size_t size() const { return data_.size(); }
  };
  
  // a set of utilities to evaluate a PDE
  template <unsigned int M, unsigned int N, unsigned int R, typename SearchEngine = ADT<M,N,R>>
  class Evaluator{
  public:
    // constructor
    Evaluator() = default;

    // produce a raster image of a general field written with respect to a LagrangianBasis defined over the given mesh.
    // the field is passed as the vector of coefficients [w_1 w_2 ... w_n] in the finite expansion \sum_{i=1}^n (w_i*psi_i(x)) which
    // writes the field as a finite linear combination of basis function.
    Raster<N> toRaster(const Mesh<M,N,R>& mesh, const DVector<double>& coeff, double h) const;
    // produce a raster image of the solution of a PDE over its domain using the specified resolution h
    template <typename E, typename F, typename B, typename I, typename S>
    Raster<N> toRaster(const PDE<M,N,R,E,F,B,I,S>& pde, double h) const;
    // produce a raster image of a callable (assuming the same interface of ScalarField)
    template <typename F>
    typename std::enable_if<std::is_invocable<F, SVector<M>>::value, Raster<N>>::type
    toRaster(const Mesh<M,N,R>& mesh, const F& f, double h) const;
  };

  // produce raster of field written as linear combination of finite element basis functions
  template <unsigned int M, unsigned int N, unsigned int R, typename SearchEngine>
  template <typename F>
  typename std::enable_if<std::is_invocable<F, SVector<M>>::value, Raster<N>>::type
  Evaluator<M,N,R,SearchEngine>::toRaster(const Mesh<M,N,R>& domain, const F& f, double h) const{
    // build search engine
    SearchEngine engine(domain);
    // define mesh domain limits
    std::array<std::pair<double, double>, N> domainRange = domain.range();
    SVector<N> p{}; // evaluation point
    double totalSize = 1; // total number of pixels in the raster
    for(std::size_t i = 0; i < N; ++i){
      p[i] = domainRange[i].first;
      totalSize *= std::ceil((domainRange[i].second - domainRange[i].first)/h) + 1;
    }
    // preallocate memory to avoid useless reallocations
    std::vector<double> data{};
    data.resize(totalSize);;
    std::vector<SVector<N>> coords{};
    coords.resize(totalSize);
    
    // start evaluation
    for(std::size_t i = 0; i < totalSize; ++i){
      // search element containing point
      std::shared_ptr<Element<M,N,R>> e = engine.search(p);
      double v = std::numeric_limits<double>::quiet_NaN();
      if(e != nullptr)
	v = f(p);
      // store result
      data[i] = v;
      coords[i] = p;
      // update evaluation point
      for(std::size_t j = 0; j < N; ++j){
	p[j] += h;
	if(p[j] > domainRange[j].second + 0.5*h){
	  p[j] = domainRange[j].first;
	}else{
	  p[j] = std::min(domainRange[j].second, p[j]);
	  break;
	};
      }
    }
    // return raster image
    return Raster<N>(data, coords, h);
  }
  
  // produce raster of field written as linear combination of finite element basis functions
  template <unsigned int M, unsigned int N, unsigned int R, typename SearchEngine>
  Raster<N> Evaluator<M,N,R,SearchEngine>::toRaster(const Mesh<M,N,R>& domain, const DVector<double>& coeff, double h) const{
    // build search engine
    SearchEngine engine(domain);
    // define mesh domain limits
    std::array<std::pair<double, double>, N> domainRange = domain.range();
    SVector<N> p{}; // evaluation point
    double totalSize = 1; // total number of pixels in the raster
    for(std::size_t i = 0; i < N; ++i){
      p[i] = domainRange[i].first;
      totalSize *= std::ceil((domainRange[i].second - domainRange[i].first)/h) + 1;
    }
    // preallocate memory to avoid useless reallocations
    std::vector<double> data{};
    data.resize(totalSize);;
    std::vector<SVector<N>> coords{};
    coords.resize(totalSize);
    
    // start evaluation
    for(std::size_t i = 0; i < totalSize; ++i){
      // search element containing point
      std::shared_ptr<Element<M,N,R>> e = engine.search(p);
      // compute value of field at point
      double v = std::numeric_limits<double>::quiet_NaN();
      if(e != nullptr){
	v = 0;
	// build a lagrangian basis over the element
	LagrangianBasis<M,N,R> basis(*e);
	// evaluate the solution at point
	for(size_t j = 0; j < ct_nnodes(M,R); ++j){
	  v += coeff[e->nodeIDs()[j]] * basis[j](p);
	}
      }
      // store result
      data[i] = v;
      coords[i] = p;
      // update evaluation point
      for(std::size_t j = 0; j < N; ++j){
	p[j] += h;
	if(p[j] > domainRange[j].second + 0.5*h){
	  p[j] = domainRange[j].first;
	}else{
	  p[j] = std::min(domainRange[j].second, p[j]);
	  break;
	};
      }
    }
    // return raster image
    return Raster<N>(data, coords, h);
  }

  // produce raster of PDE solution
  template <unsigned int M, unsigned int N, unsigned int R, typename SearchEngine>
  template <typename E, typename F, typename B, typename I, typename S>
  Raster<N> Evaluator<M,N,R,SearchEngine>::toRaster(const PDE<M,N,R,E,F,B,I,S>& pde, double h) const {
    return toRaster(pde.domain(), pde.solution(), h);
  }
  
}}}
#endif // __EVALUATOR_H__
