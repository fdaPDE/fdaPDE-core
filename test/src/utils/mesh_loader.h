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

#ifndef __MESH_LOADER_H__
#define __MESH_LOADER_H__

#include <fdaPDE/geometry.h>
#include <fdaPDE/utils.h>
#include <gtest/gtest.h>   // testing framework
#include <random>
using fdapde::core::CSVReader;
#include "utils.h"

namespace fdapde {
namespace testing {
  
using MESH_TYPE_LIST = ::testing::Types<
  core::Triangulation<2,2>/*, core::Triangulation<2,3>, core::Triangulation<3,3>*//*, core::NetworkMesh*/>;

template <int LocalDim, int EmbedDim>
core::Triangulation<LocalDim, EmbedDim> read_triangulation(const std::string& path) {
    DMatrix<double> nodes = read_csv<double>(path + "/points.csv");
    DMatrix<int> cells = (read_csv<int>(path + "/elements.csv").array() - 1).matrix();   // realign indexes to 0
    DMatrix<int> boundary = read_csv<int>(path + "/boundary.csv");
    // initialize mesh
    return core::Triangulation<LocalDim, EmbedDim>(nodes, cells, boundary);
}

    // // generate element at random inside mesh m
    // typename MeshType::CellType generate_random_element() {
    //     std::uniform_int_distribution<int> random_id(0, mesh.n_cells() - 1);
    //     int id = random_id(rng);
    //     return mesh.cell(id);
    // };
    // // generate point at random inside element e
    // SVector<N> generate_random_point(const typename MeshType::CellType& e) {
    //     std::uniform_real_distribution<double> T(0, 1);
    //     // let t, s, u ~ U(0,1) and P1, P2, P3, P4 a set of points, observe that:
    //     //     * if P1 and P2 are the vertices of a linear element, p = t*P1 + (1-t)*P2 lies into it for any t ~ U(0,1)
    //     //     * if P1, P2, P3 are vertices of a triangle, the point P = (1-t)P1 + t((1-s)P2 + sP3) is in the triangle
    //     //       for any choice of t, s ~ U(0,1)
    //     //     * if P1, P2, P3, P4 are vertices of a tetrahedron, then letting Q = (1-t)P1 + t((1-s)P2 + sP3) and
    //     //       P = (1-u)P4 + uQ, P belongs to the tetrahedron for any choice of t, s, u ~ U(0,1)
    //     double t = T(rng);
    //     SVector<N> p = t * e.node(0) + (1 - t) * e.node(1);
    //     for (int j = 1; j < M; ++j) {
    //         t = T(rng);
    //         p = (1 - t) * e.node(1 + j) + t * p;
    //     }
    //     return p;
    // }
    // // generate randomly n pairs <ID, point> on mesh, such that point is contained in the element with identifier ID
    // std::vector<std::pair<int, SVector<N>>> sample(int n) {
    //     // preallocate memory
    //     std::vector<std::pair<int, SVector<N>>> result {};
    //     result.resize(n);
    //     // generate sample
    //     for (int i = 0; i < n; ++i) {
    //         auto e = generate_random_element();
    //         result[i] = std::make_pair(e.id(), generate_random_point(e));
    //     }
    //     return result;
    // }

  
}   // namespace testing
}   // namespace fdapde

#endif   // __MESH_LOADER_H__
