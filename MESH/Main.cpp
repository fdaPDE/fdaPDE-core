#include <array>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <set>
#include <Eigen/QR>

#include "CSVReader.h"
#include "Mesh.h"
#include "engines/BruteForce/BruteForce.h"
#include "engines/BarycentricWalk/BarycentricWalk.h"
#include "../utils/DataStructures/Tree.h"
#include "engines/AlternatingDigitalTree/ADT.h"
#include "Geometry.h"

#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::microseconds;

//EIGEN_MAKE_ALIGNED_OPERATOR_NEW

using namespace fdaPDE::core::MESH;

#include <random>

typedef std::mt19937 RNG;  // the Mersenne Twister with a popular choice of parameters

int main(void) {

  //Mesh<2,2> m("points_2.csv", "edges_2.csv", "triangles_2.csv", "neighbors_2.csv");

  //Mesh<2,2> m("points.csv", "edges.csv", "triangles.csv", "neighbors.csv");

  Mesh<2,2> m("circle_points.csv", "circle_edges.csv", "circle_triangles.csv", "circle_neighbors.csv", "circle_boundary_markers.csv");
  
  std::cout << "mesh information loaded" << std::endl;
  
  std::cout << m.getMeshRange()[0].first << " - " << m.getMeshRange()[0].second << std::endl;
  std::cout << m.getMeshRange()[1].first << " - " << m.getMeshRange()[1].second << std::endl;
  std::cout << m.getMeshRange()[2].first << " - " << m.getMeshRange()[2].second << std::endl;
  
  //std::cout << triangles.toEigen() << std::endl;

  //  for(std::shared_ptr<Element> e : m){
    
  //}

  // std::shared_ptr<Element<2,2>> e = m.requestElementById(50);
  // std::cout << e->getCoords()[0] << std::endl;
  // std::cout << e->getCoords()[1] << std::endl;
  // std::cout << e->getCoords()[2] << std::endl;
  // std::cout << "-----------------" << std::endl;
  
  // std::cout << e->computeBarycentricCoordinates(SVector<2>(-2.0,-1.75)) << std::endl;
  // std::cout << e->contains(SVector<2>(-2.0,-1.75)) << std::endl;

  // std::cout << "----------------- cose da albero --------------------" << std::endl;
  // Tree<std::string> t("sono la radice");
  // std::cout << t.getNode(0)->getData() << std::endl;

  // // inseriamo un nodo??
  // t.insert("sono il primo nodo :)");
  // std::cout << t.getNode(1)->getData() << std::endl;

  // // chi sono i figli del padre??
  // for(auto figli : t.getNode(0)->getChildren()){
  //   if(figli != nullptr){
  //     std::cout << figli->getKey() << std::endl;
  //   }
  // }

  // t.insert("sono il fratello del figlio della radice :D");

  // // chi sono i figli del padre??
  // for(auto figli : t.getNode(0)->getChildren()){
  //   if(figli != nullptr){
  //     std::cout << figli->getKey() << std::endl;
  //     std::cout << figli->getData() << std::endl;
  //   }
  // }

  // t.insert("nessuno mi caga... :D");

  // // chi sono i figli del padre??
  // for(auto figli : t.getNode(0)->getChildren()){
  //   if(figli != nullptr){
  //     std::cout << figli->getKey() << std::endl;
  //     std::cout << figli->getData() << std::endl;
  //   }
  // }
  
  // std::cout << t.getNode(1)->isLeaf() << std::endl;

  // costruiamo un ADT??

  // definiamo un pò di punti... facciamo 5
  // SVector<2> punto1(0.02, 0.7);
  // SVector<2> punto2(0.2, 0.3);
  // SVector<2> punto3(0.7, 0.45);
  // SVector<2> punto4(0.23, 0.9);
  // SVector<2> punto5(0.4, 0.6);

  // costruiamo un ADT...
  // std::vector<SVector<2>> data = {punto1, punto2, punto3, punto4, punto5};
  // TreeSearch<2> ADT;
  // ADT.init(data);

  
  // std::cout << "ADT built" << std::endl;
  // std::cout << ADT.getTree().getNumberOfNodes() << std::endl;

  // std::cout << "radice" << std::endl;
  // std::cout << ADT.getTree().getNode(0)->getData().getPoint() << std::endl;

  // std::cout << "figlio sinistro" << std::endl;
  // std::cout << ADT.getTree().getNode(1)->getData().getPoint() << std::endl;

  // std::cout << "figlio destro" << std::endl;
  // std::cout << ADT.getTree().getNode(2)->getData().getPoint() << std::endl;

  // std::cout << "figlio destro di figlio sinistro" << std::endl;
  // std::cout << ADT.getTree().getNode(3)->getData().getPoint() << std::endl;

  // std::cout << "figlio destro di figlio destro di figlio sinistro" << std::endl;
  // std::cout << ADT.getTree().getNode(4)->getData().getPoint() << std::endl;
  // std::cout << "(" << ADT.getTree().getNode(4)->getData().getRange().first << ", " << ADT.getTree().getNode(4)->getData().getRange().second << ")" << std::endl;

  // // proviamo una query :D

  // rectangle<2> query = std::make_pair(SVector<2>(0,0.55), SVector<2>(0.45,1));
  // std::list<SVector<2>> result = ADT.search(query);
  
  // std::cout << "risultati query....... :X" << std::endl;
  // std::cout << result.size() << std::endl;

  // for(SVector<2> res : result){
  //   std::cout << "------" << std::endl;  
  //   std::cout << res << std::endl;
  // }

  // specific members of the engine

  // ############
  uint32_t seed;
  RNG rng;
  std::uniform_real_distribution<double> uniform_real;

  seed = time(NULL);  // seed for RNG
  rng  = RNG(seed);   // define RNG
  
  // define uniform distribution over the ID space
  uniform_real = std::uniform_real_distribution<double>(-0.95,0.95); // radius
  
  constexpr int NN = 10000;
  std::vector<std::pair<double, double>> tests;
  std::vector<double> test3D;
  tests.reserve(NN);
  test3D.reserve(NN);

  for(size_t j = 0; j < NN; ++j){
    bool inserted = false;
    do{
      double x1 = uniform_real(rng);
      double x2 = uniform_real(rng);
      double x3 = uniform_real(rng);
      if(std::pow(x1,2) + std::pow(x2,2) + std::pow(x3,2) < std::pow(0.95, 2)){
	tests.push_back(std::make_pair(x1,x2));
	test3D.push_back(x3);
	inserted = true;
    }
    }while(inserted != true);
  }

  std::array<unsigned int, NN> resultBF;
  std::array<unsigned int, NN> resultADT;
  std::array<unsigned int, NN> resultWalk;
  
  std::cout << "----------------- bruteforce" << std::endl;
  BruteForce<2,2> bruteForceEngine(m);
  auto t1 = high_resolution_clock::now();
  for(size_t j = 0; j < NN; ++j){
    resultBF[j] = bruteForceEngine.search(SVector<2>(tests[j].first,tests[j].second))->getID();
  }
  auto t2 = high_resolution_clock::now();

  auto ms_int = duration_cast<microseconds>(t2 - t1);
  std::cout << "duration:                             " << ms_int.count() << "us" << std::endl;
  std::cout << "average time:                         " << ms_int.count()/NN << "us" << std::endl;
  
  std::cout << "----------------- barycentric walk" << std::endl;
  BarycentricWalk<2,2> walkEngine(m);
  t1 = high_resolution_clock::now();
  for(size_t j = 0; j < NN; ++j){
    resultWalk[j] = walkEngine.search(SVector<2>(tests[j].first,tests[j].second))->getID();
  }  
  t2 = high_resolution_clock::now();
  
  ms_int = duration_cast<microseconds>(t2 - t1);
  std::cout << "duration:                             " << ms_int.count() << "us" << std::endl;
  std::cout << "average time:                         " << ms_int.count()/NN << "us" << std::endl;

  std::cout << "----------------- ADT" << std::endl;
  ADT<2,2> treeEngine(m);
  t1 = high_resolution_clock::now();
  for(size_t j = 0; j < NN; ++j){
    resultADT[j] = treeEngine.search(SVector<2>(tests[j].first,tests[j].second))->getID();
  }  
  t2 = high_resolution_clock::now();
  
  ms_int = duration_cast<microseconds>(t2 - t1);
  std::cout << "duration:                             " << ms_int.count() << "us" << std::endl;
  std::cout << "average time:                         " << ms_int.count()/NN << "us" << std::endl;

  std::cout << "----------------- check correctness" << std::endl;
  bool correct = true;
  for(size_t j = 0; j < NN; ++j){
    if((resultBF[j] != resultADT[j]) || (resultBF[j] != resultWalk[j]) || (resultADT[j] != resultWalk[j])){
      correct = false;
      break;
    }
  }
  if(correct == true){
    std::cout << "OK: all queries match" << std::endl;
  }
  else{
    std::cout << "FAILED" << std::endl;
  }
  
  // we have now access to mesh information...

  // std::cout << "some geometric experiments..." << std::endl;

  // constexpr unsigned N = 2;

  // SVector<N> a(1,2);
  
  // std::vector<SVector<N>> inputBasis = {a};
  
  // std::vector<SVector<N>> orthonormalBasis = Geometry<N>::orthonormalize(inputBasis);
  
  // for(SVector<N> vect : orthonormalBasis){
  //   std::cout << vect << std::endl;
  // }

  // std::cout << "project some point..." << std::endl;

  // SVector<N> projectedOnPlane(3,-8);

  // SVector<N> proj = Geometry<N>::projectInto(orthonormalBasis, projectedOnPlane);
  
  // std::cout << Geometry<N>::projectOnto(orthonormalBasis, projectedOnPlane) << std::endl;
  // std::cout << "...." << std::endl;
  // std::cout << Geometry<N>::projectInto(orthonormalBasis, projectedOnPlane) << std::endl;
  // std::cout << ".... compute distance" << std::endl;
  // std::cout << Geometry<N>::getL2Distance(orthonormalBasis, SVector<N>::Zero(), proj) << std::endl;

  // // esperimenti 2.5D
  // std::cout << "manifold 2.5D" << std::endl;
  // Mesh<2,3> d("nodes_2.5D.csv", "edges_2.5D.csv", "triangles_2.5D.csv", "neighbors_2.5D.csv");

  // std::cout << "data loaded" << std::endl;
  // std::cout << d.requestElementById(5)->computeBarycentricCoordinates(SVector<3>(1,1,1)) << std::endl;

  // std::cout << d.requestElementById(5)->getCoords()[0] << std::endl;
  // std::cout << ".........." << std::endl;
  // std::cout << d.requestElementById(5)->getCoords()[1] << std::endl;
  // std::cout << ".........." << std::endl;
  // std::cout << d.requestElementById(5)->getCoords()[2] << std::endl;
  // std::cout << ".........." << std::endl;
  
  // std::array<SVector<3>, 3> coords = d.requestElementById(5)->getCoords();
  
  // SVector<3> a1 = (coords[1] - coords[0]);
  // SVector<3> b1 = (coords[2] - coords[0]);
  
  // std::cout << a1 << std::endl;
  // std::cout << b1 << std::endl;
  // std::cout << ".........." << std::endl;

  // std::cout << std::numeric_limits<double>::epsilon() << std::endl;
  
  // for(SVector<3> vect : Geometry<3>::orthonormalize({a1,b1})){
  //   std::cout << "ci sono" << std::endl;
  //   std::cout << vect << std::endl;
  // }

  // // select a point which by construction is inside the element (plane passing by point 5)
  // std::function<double(double,double)> plane = [](double x, double y) -> double {
  //   return (0.00581206944*x + 0.0023074181*y + 0.0096373185234446)/(0.01276304744);
  // };
  
  // // check if is inside the element.... should return true
  // std::cout << d.requestElementById(5)->contains(SVector<3>(0.37, 0.1, plane(0.37,0.1))) << std::endl;
  
  // ADTSearch<2,3> treeEngine(d);
  
  // // should return 5... and indeed is so :)
  // std::cout << treeEngine.search(SVector<3>(0.37, 0.1, plane(0.37,0.1)))->getID() << std::endl;
  
  return 0;
}
