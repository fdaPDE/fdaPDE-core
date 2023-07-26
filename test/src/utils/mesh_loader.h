#ifndef __MESH_LOADER_H__
#define __MESH_LOADER_H__

#include <gtest/gtest.h> // testing framework
#include <random>

#include <fdaPDE/utils.h>
#include <fdaPDE/mesh.h>
using fdapde::core::NetworkMesh;
using fdapde::core::Mesh2D;
using fdapde::core::Mesh3D;
using fdapde::core::SurfaceMesh;
using fdapde::core::is_linear_network;
using fdapde::core::is_manifold;
using fdapde::core::Element;
#include "../../../fdaPDE/utils/IO/CSVReader.h"

namespace fdapde {
namespace testing {
  
  const std::string MESH_PATH = "../data/mesh/";
  using MESH_TYPE_LIST = ::testing::Types<Mesh2D<>, SurfaceMesh<>, Mesh3D<>, NetworkMesh<>>;

  // selects sample mesh depending on the dimensionality of the problem
  //     * 1.5D: 204  2D points, 559   elements, 559   edges. /test/data/mesh/linear_newtwork/*.csv
  //     * 2D:   3600 2D points, 6962  elements, 10561 edges. /test/data/mesh/unit_square/*.csv
  //     * 2.5D: 340  3D points, 616   elements, 956   edges. /test/data/mesh/surface/*.csv
  //     * 3D:   587  3D points, 2775  elements, 5795  faces. /test/data/mesh/unit_sphere/*.csv
  constexpr const auto standard_mesh_selector(unsigned int M, unsigned int N) {
    // first order meshes
    if(M == 1 && N == 2) return "network";     // 1.5D
    if(M == 2 && N == 2) return "unit_square"; // 2D
    if(M == 2 && N == 3) return "surface";     // 2.5D
    if(M == 3 && N == 3) return "unit_sphere"; // 3D
    return ""; // error case
  }
  
  // An utility class to help in the import of sample test meshes from files 
  template <typename E>
  struct MeshLoader {
    E mesh;
    // expose the dimensionality of the mesh
    static constexpr unsigned int M = E::local_dimension;
    static constexpr unsigned int N = E::embedding_dimension;
    static constexpr unsigned int R = E::order;
    static constexpr bool manifold = is_manifold<M, N>::value;
    
    CSVReader<double> Dreader{}; CSVReader<int> Ireader; // csv parser
    // raw files
    CSVFile<double> pointsCSV;
    CSVFile<int> elementsCSV;
    CSVFile<int> edgesCSV;
    CSVFile<int> boundaryCSV;
    // cope with different storage strategies adopted by linear network meshes
    typename std::conditional<
      !is_linear_network<M, N>::value, CSVFile<int>, CSVSparseFile<int>
      >::type neighCSV;

    // RNG for generation of random elements and points in mesh
    std::random_device rng;

    // constructors
    MeshLoader(const std::string& meshID){
      std::string point    = MESH_PATH + meshID + "/points.csv";
      std::string edges    = MESH_PATH + meshID + "/edges.csv";
      std::string elements = MESH_PATH + meshID + "/elements.csv";
      std::string neigh    = MESH_PATH + meshID + "/neigh.csv";
      std::string boundary = MESH_PATH + meshID + "/boundary.csv";
      
      // initialize test objects
      mesh = E(point, elements, neigh, boundary);

      // load raw data
      pointsCSV   = Dreader.parseFile(point);
      elementsCSV = Ireader.parseFile(elements);
      edgesCSV    = Ireader.parseFile(edges);
      boundaryCSV = Ireader.parseFile(boundary);
      // proper parse the neighboring information in case of 1.5D meshes
      if constexpr(!is_linear_network<M, N>::value)  
	neighCSV = Ireader.parseFile(neigh);
      else
	neighCSV = Ireader.parseSparseFile(neigh);
    }
    // load default mesh according to dimensionality
    MeshLoader() : MeshLoader(standard_mesh_selector(E::local_dimension, E::embedding_dimension)) {};
    
    // some usefull utilities for testing
    
    // generate element at random inside mesh m
    const Element<E::local_dimension, E::embedding_dimension, E::order>& generateRandomElement();
    // generate point at random inside element e
    SVector<E::embedding_dimension> generateRandomPoint(
    const Element<E::local_dimension, E::embedding_dimension, E::order>& e);

    // generate randomly n pairs <ID, point> on mesh, such that point is contained in the element with identifier ID
    std::vector<std::pair<std::size_t, SVector<E::embedding_dimension>>> sample(std::size_t n);
  };
  
  template <typename E>
  const Element<E::local_dimension, E::embedding_dimension, E::order>&
  MeshLoader<E>::generateRandomElement() {
    std::uniform_int_distribution<int> randomID(0, mesh.elements()-1);
    int ID = randomID(rng);  
    return mesh.element(ID);
  }

  template <typename E>
  SVector<E::embedding_dimension> MeshLoader<E>::generateRandomPoint(
  const Element<E::local_dimension, E::embedding_dimension, E::order>& e) {
    std::uniform_real_distribution<double> T(0,1);
    // let t, s, u ~ U(0,1) and P1, P2, P3, P4 a set of points, observe that:
    //     * if P1 and P2 are the vertices of a linear element, p = t*P1 + (1-t)*P2 lies into it for any t ~ U(0,1)
    //     * if P1, P2, P3 are vertices of a triangle, the point P = (1-t)P1 + t((1-s)P2 + sP3) is in the triangle
    //       for any choice of t, s ~ U(0,1)
    //     * if P1, P2, P3, P4 are vertices of a tetrahedron, then letting Q = (1-t)P1 + t((1-s)P2 + sP3) and
    //       P = (1-u)P4 + uQ, P belongs to the tetrahedron for any choice of t, s, u ~ U(0,1)
    double t = T(rng);
    SVector<N> p = t*e.coords()[0] + (1-t)*e.coords()[1];
    for(std::size_t j = 1; j < M; ++j){
      t = T(rng);
      p = (1-t)*e.coords()[1+j] + t*p;
    }
    return p;
  }

  template <typename E>
  std::vector<std::pair<std::size_t, SVector<E::embedding_dimension>>> MeshLoader<E>::sample(std::size_t n) {
    // preallocate memory
    std::vector<std::pair<std::size_t, SVector<E::embedding_dimension>>> result{};
    result.resize(n);
    // generate sample
    for(std::size_t i = 0; i < n; ++i){
      auto e = generateRandomElement();
      SVector<E::embedding_dimension> p = generateRandomPoint(e);
      result[i] = std::make_pair(e.ID(), p);
    }
    return result;
  }
  
}}

#endif // __MESH_LOADER_H__
