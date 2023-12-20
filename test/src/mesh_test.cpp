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

#include <fdaPDE/mesh.h>
#include <fdaPDE/utils.h>
#include <gtest/gtest.h>   // testing framework

#include <memory>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>
using fdapde::core::Element;
using fdapde::core::Mesh;

#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MESH_TYPE_LIST;
using fdapde::testing::MeshLoader;

// test suite for testing both non-manifold meshes (2D/3D) and manifold mesh (2.5D/1.5D)
template <typename E> struct mesh_test : public ::testing::Test {
    MeshLoader<E> mesh_loader {};   // use default mesh
    static constexpr unsigned int M = MeshLoader<E>::M;
    static constexpr unsigned int N = MeshLoader<E>::N;
    typedef E MeshType;
};
TYPED_TEST_SUITE(mesh_test, MESH_TYPE_LIST);

// check points' coordinate embedded in an element are loaded correctly
TYPED_TEST(mesh_test, elements_construction) {
    for (std::size_t i = 0; i < this->mesh_loader.mesh.n_elements(); ++i) {
        // request element with ID i
        auto e = this->mesh_loader.mesh.element(i);
        // check coordinates stored in element built from Mesh object match raw informations
        int j = 0;
        auto raw_elements = this->mesh_loader.elements_.row(i);
        auto raw_points = this->mesh_loader.points_;
        for (int k = 0; k < raw_elements.size(); ++k) {
            int nodeID = raw_elements[k];
            auto p = raw_points.row(nodeID);
            SVector<TestFixture::N> ePoint = e.coords()[j];
            for (std::size_t idx = 0; idx < TestFixture::N; ++idx) { EXPECT_TRUE(almost_equal(p[idx], ePoint[idx])); }
            j++;
        }
    }
}

// check edges informations are computed correctly (up to an ordering of the nodes)
TYPED_TEST(mesh_test, edges_construction) {
    constexpr int K = TestFixture::MeshType::n_vertices_per_facet;
    // load raw edges
    std::vector<std::vector<int>> expected_edge_set;
    for (std::size_t i = 0; i < this->mesh_loader.edges_.rows(); ++i) {
        std::vector<int> e {};
        for (std::size_t j = 0; j < K; ++j) { e.push_back(this->mesh_loader.edges_(i, j)); }
        std::sort(e.begin(), e.end());   // normalize wrt ordering of edge's nodes
        expected_edge_set.push_back(e);
    }
    // load mesh edges and compute mask
    std::vector<std::vector<int>> mesh_edge_set;
    std::vector<bool> edge_mask(expected_edge_set.size(), false);
    for (auto it = this->mesh_loader.mesh.facet_begin(); it != this->mesh_loader.mesh.facet_end(); ++it) {
        std::vector<int> e {};
        for (std::size_t j = 0; j < K; ++j) { e.push_back((*it).node_ids()[j]); }
        std::sort(e.begin(), e.end());   // normalize wrt ordering of edge's node

        // find this edge in expected set
        auto search_it = std::find(expected_edge_set.begin(), expected_edge_set.end(), e);
        if (search_it != expected_edge_set.end()) {
            edge_mask[std::distance(expected_edge_set.begin(), search_it)] = true;
        }
    };
    // check all expected edges are indeed computed
    bool result = true;
    for (bool b : edge_mask) { result &= b; }
    EXPECT_TRUE(result == true);
}

// check neighbors informations are computed correctly (up to a permutation of the elements)
TYPED_TEST(mesh_test, neighbors_construction) {
    // same number of elements
    EXPECT_TRUE(this->mesh_loader.neighbors_.size() == this->mesh_loader.mesh.neighbors().size());

    if constexpr (!fdapde::core::is_network<TestFixture::M, TestFixture::N>::value) {
        bool result = true;
	int n_row = this->mesh_loader.neighbors_.rows();
	int n_col = this->mesh_loader.neighbors_.cols();
	for(std::size_t i = 0; i < n_row; ++i) {
	  for(std::size_t j = 0; j < n_col; ++j){
	    if(this->mesh_loader.neighbors_(i,j) != this->mesh_loader.mesh.neighbors()(i,j)) { result = false; }
	  }
	}
        EXPECT_TRUE(result == true);
    } else {
      // for linear networks, neighbors are stored as a sparse adjacency matrix
      bool result = true;
      for (int k = 0; k < this->mesh_loader.neighbors_.outerSize(); ++k) {
	for (SpMatrix<int>::InnerIterator it(this->mesh_loader.neighbors_,k); it; ++it) {
	  if(it.value() != this->mesh_loader.mesh.neighbors().coeff(it.row(), it.col())) { result = false; }
	}
      }
      EXPECT_TRUE(result == true);
    }
}

// performs some checks on the mesh topology, e.g. checks that stated neighbors shares exactly M points
TYPED_TEST(mesh_test, boundary_checks) {
    // cycle over all mesh elements
    for (std::size_t i = 0; i < this->mesh_loader.mesh.n_elements(); ++i) {
        auto e = this->mesh_loader.mesh.element(i);
        // check that neighboing elements have always M points in common
        for (int neigh_id : e.neighbors()) {
            if (!e.is_on_boundary()) {
                // request neighboring element from mesh
                auto n = this->mesh_loader.mesh.element(neigh_id);
                // take nodes of both elements
                std::array<SVector<TestFixture::N>, TestFixture::M + 1> eList, nList;
                for (std::size_t j = 0; j < TestFixture::M + 1; ++j) {
                    eList[j] = e.coords()[j];
                    nList[j] = n.coords()[j];
                }
                // check that the points in common between the two are exactly M
                std::size_t matches = 0;
                for (SVector<TestFixture::N> p : eList) {
                    if (std::find(nList.begin(), nList.end(), p) != nList.end()) matches++;
                }
		EXPECT_TRUE(matches == TestFixture::M);
            } else {
                // check that at least one vertex of e is detected as boundary point
                bool element_on_boundary = false;
                auto node_ids = e.node_ids();
                for (std::size_t n : node_ids) {
                    if (this->mesh_loader.mesh.is_on_boundary(n)) {   // mesh detects this point as boundary point
                        element_on_boundary = true;
                    }
                }
                EXPECT_TRUE(element_on_boundary);
            }
        }
    }
}

// check the range for loop scans the whole mesh element by element
TYPED_TEST(mesh_test, range_for) {
    // prepare set with all indexes of IDs to touch
    std::unordered_set<int> mesh_ids {};
    for (int i = 0; i < this->mesh_loader.mesh.n_elements(); ++i) mesh_ids.insert(i);

    // range-for over all elements removing the element's ID from the above set when the element is visited
    for (const auto& e : this->mesh_loader.mesh) {
        // check element ID still present in the IDs set (ID not visisted by means of a different element)
        EXPECT_TRUE(mesh_ids.find(e.ID()) != mesh_ids.end());
        mesh_ids.erase(e.ID());
    }
    // check that no ID is left in the initial set
    EXPECT_TRUE(mesh_ids.empty());
}

TEST(mesh_test, 1D_interval) {
    // create unit interval (nodes evenly distributed)
    Mesh<1, 1> unit_interval(0, 1, 10);

    std::unordered_set<int> mesh_ids {};
    for (int i = 0; i < unit_interval.n_elements(); ++i) { mesh_ids.insert(i); }
    for (const auto& e : unit_interval) {
        // check element ID still present in the IDs set (ID not visisted by means of a different element)
        EXPECT_TRUE(mesh_ids.find(e.ID()) != mesh_ids.end());
        mesh_ids.erase(e.ID());

	// boundary checks
        if (e.ID() == 0 || e.ID() == unit_interval.n_elements()-1) {
            EXPECT_TRUE(e.is_on_boundary());
        } else {
	    EXPECT_TRUE(!e.is_on_boundary());
        }
    }
    // check that no ID is left in the initial set
    EXPECT_TRUE(mesh_ids.empty());   
}
