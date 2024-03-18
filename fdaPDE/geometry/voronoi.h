// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __VORONOI_H__
#define __VORONOI_H__

#include <unordered_set>
#include "../utils/compile_time.h"
#include "../utils/symbols.h"
#include "mesh.h"

namespace fdapde {
namespace core {

// computes the voronoi diagram of a delanoy triangulation

// compute the circumcircle's center of a 2D triangle
// see https://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
constexpr SVector<2> circumcenter(const SVector<2>& p1, const SVector<2>& p2, const SVector<2>& p3) {
    // load vertices in cpu registers
    double x1 = p1[0], y1 = p1[1];
    double x2 = p2[0], y2 = p2[1];
    double x3 = p3[0], y3 = p3[1];

    double y2y1 = y2 - y1;
    double y3y1 = y3 - y1;
    double x2x1 = x2 - x1;
    double x3x1 = x3 - x1;

    double D = 2 * (x2x1 * y3y1 - x3x1 * y2y1);

    double s1 = x1 * x1 + y1 * y1;
    double s2 = x2 * x2 + y2 * y2;
    double s3 = x3 * x3 + y3 * y3;
    // return circumcenter coordinates
    return {((s2 - s1) * y3y1 - (s3 - s1) * y2y1) / D, ((s3 - s1) * x2x1 - (s2 - s1) * x3x1) / D};
}
constexpr SVector<2> circumcenter(const Element<2, 2>& e) { return circumcenter(e.coords()[0], e.coords()[1], e.coords()[2]); }

// compute measure of 2D triangle given its vertices
constexpr double measure(const SVector<2>& p1, const SVector<2>& p2, const SVector<2>& p3) {
    return 0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]));
}

// // compute the circumcirles's center of a 3D tetrahedron
// SVector<3> circumcenter(const SVector<2>& p1, const SVector<2>& p2, const SVector<2>& p3, const SVector<2>& p4) {
//     // load vertices in cpu registers
//     double x1 = p1[0], y1 = p1[1];
//     double x2 = p2[0], y2 = p2[1];
//     double x3 = p3[0], y3 = p3[1];
//     double x4 = p4[0], 

// }

// compute the circumsphere's center of a 3D tetrahedron

// same goes for segments and surface triangles

  // dual of a Delanoys triangulation
template <int M, int N>
class Voronoi {
private:
  const Mesh<M, N>* mesh_;
  // centers of voronoi cells are equal to nodes of the mesh_
  DMatrix<double> sites_;
  int num_nodes_ = 0;

  DMatrix<int> edges_; // for each node, the node to which it is connected
  BinaryVector<fdapde::Dynamic> boundary_; // i-th element true if i-th node is on boundary
  DVector<double> measures_; // measure of the i-th Voronoi cell
  
public:

  // on a constrained voronoi there is a number of nodes = number of elements + number of boundary faces

  int n_elements() const { return mesh_->n_nodes(); }
  
  Voronoi() = default;
  Voronoi(const Mesh<M, N>& mesh) : mesh_(&mesh) {   // constructs voronoi diagram from Delanoy triangulation
      static constexpr int n_vertices_per_facet = M;
      // compute nodes
      nodes_.resize(mesh.n_elements(), N);

      auto edge_pattern = combinations<M, Mesh<M, N>::n_facets_per_element>();
      std::unordered_set<std::array<int, M>, std_array_hash<int, M>> edge_set;
      std::array<int, M> edge;

      for (const auto& e : mesh) {   // the nodes of a voronoi diagram are the circumcenter of the elements
          nodes_.row(e.ID()) = circumcenter(e);

          for (std::size_t i = 0; i < edge_pattern.rows(); ++i) {
              for (std::size_t j = 0; j < n_vertices_per_facet; ++j) {
                  edge[j] = mesh.neighbors()(e.ID(), edge_pattern(i, j));
              }
              std::sort(edge.begin(), edge.end());   // avoid duplicated edges (sorting will cause hash collision)
              edge_set.insert(edge);
          }
	  // if element is on boundary, add boundary node and corresponding edge
      }

      std::cout << nodes_ << std::endl;

      // move edges to matrix
      edges_.resize(edge_set.size(), M);
      std::size_t row = 0;
      for (const auto& edge : edge_set) {
          for (std::size_t i = 0; i < M; ++i) { edges_(row, i) = edge[i]; }
          row++;
      }

      // std::cout << edges_ << std::endl;
  }

  DVector<int> locate()
  
};

}   // namespace core
}   // namespace fdapde

#endif   // __VORONOI_H__
