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

#ifndef __LINEAR_NETWORK_H__
#define __LINEAR_NETWORK_H__

#include "segment.h"
#include "../linear_algebra/binary_matrix.h"

namespace fdapde {
  
// template specialization for 1D meshes (bounded intervals)
template <int M, int N> class Triangulation;
template <> class Triangulation<1, 2> {
   public:
    static constexpr int local_dim = 1;
    static constexpr int embed_dim = 2;
    static constexpr int n_nodes_per_cell = 2;
    static constexpr int n_neighbors_per_cell = Dynamic;
    static constexpr bool is_manifold = true;
    using VertexType = SVector<2>;
    using LocationPolicy = TreeSearch<Triangulation<1, 2>>;

    struct CellType : public Segment<Triangulation<1, 2>> {
       private:
        using Base = Segment<Triangulation<1, 2>>;
        using Base::id_;
        using Base::mesh_;
       public:
        CellType() = default;
        CellType(int id, const Triangulation* mesh) : Segment<Triangulation<1, 2>>(id, mesh) { }

        DVector<int> neighbors() const {
            const auto& v1 = mesh_->node_to_cells().at(mesh_->cells()(id_, 0));
            const auto& v2 = mesh_->node_to_cells().at(mesh_->cells()(id_, 1));
            DVector<int> result(v1.size() + v2.size());
            int i = 0;
            for (; i < v1.size(); ++i) result[i] = v1[i];
            for (; i < v1.size() + v2.size(); ++i) result[i] = v2[i];
            return result;
        }
    };

    Triangulation() = default;
    Triangulation(const DMatrix<double>& nodes, const DMatrix<int>& cells, const DMatrix<int>& boundary) :
        nodes_(nodes), cells_(cells), nodes_markers_(boundary) {
        // store number of nodes and number of elements
        n_nodes_ = nodes_.rows();
        n_cells_ = cells_.rows();
        // compute mesh limits
        range_.row(0) = nodes_.colwise().minCoeff();
        range_.row(1) = nodes_.colwise().maxCoeff();
	neighbors_.resize(n_cells_, n_cells_);
	// compute node to cells boundings
	for (int i = 0; i < n_cells_; ++i) {
	  node_to_cells_[cells_(i, 0)].push_back(i);
	  node_to_cells_[cells_(i, 1)].push_back(i);
	}
	// recover adjacency matrix
	SpMatrix<short> adjoint_neighbors;
	std::vector<Eigen::Triplet<int>> triplet_list;
	for (const auto& [node, edges] : node_to_cells_) {
	  for (int i = 0; i < edges.size(); ++i) {
            for (int j = i + 1; j < edges.size(); ++j) triplet_list.emplace_back(edges[j], edges[i], 1);
	  }
	}
	adjoint_neighbors.resize(n_cells_, n_cells_);
	adjoint_neighbors.setFromTriplets(triplet_list.begin(), triplet_list.end());
	neighbors_ = adjoint_neighbors.selfadjointView<Eigen::Lower>();   // symmetrize neighboring relation
    };

    // getters
    CellType cell(int id) const { return CellType(id, this); }
    VertexType node(int id) const { return nodes_.row(id); }
    bool is_node_on_boundary(int id) const { return nodes_markers_[id]; }
    const DMatrix<double>& nodes() const { return nodes_; }
    const DMatrix<int, Eigen::RowMajor>& cells() const { return cells_; }
    const SpMatrix<short>& neighbors() const { return neighbors_; }
    const BinaryVector<fdapde::Dynamic>& boundary() const { return nodes_markers_; }
    int n_cells() const { return n_cells_; }
    int n_nodes() const { return n_nodes_; }
    SVector<2> range() const { return range_; }

    // iterators support
    class cell_iterator : public index_based_iterator<cell_iterator, CellType> {
        using Base = index_based_iterator<cell_iterator, CellType>;
        using Base::index_;
        friend Base;
        const Triangulation* mesh_;
        cell_iterator& operator()(int i) {
            Base::val_ = mesh_->cell(i);
            return *this;
        }
       public:
        cell_iterator(int index, const Triangulation* mesh) : Base(index, 0, mesh->n_cells_), mesh_(mesh) {
            if (index_ < mesh_->n_cells_) operator()(index_);
        }
    };
    cell_iterator cells_begin() const { return cell_iterator(0, this); }
    cell_iterator cells_end() const { return cell_iterator(n_cells_, this); }

    // point location
    DVector<int> locate(const DMatrix<double>& points) const {
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->locate(points);
    }
    // the set of cells which have node id as vertex
    std::vector<int> node_patch(int id) const { return node_to_cells_.at(id); }
   protected:
    DMatrix<double> nodes_;                            // physical coordinates of mesh's vertices
    DMatrix<int, Eigen::RowMajor> cells_ {};           // nodes (as row indexes in nodes_ matrix) composing each cell
    SpMatrix<short> neighbors_ {};                     // ids of faces adjacent to a given face (-1 if no adjacent face)
    BinaryVector<fdapde::Dynamic> nodes_markers_ {};   // j-th element is 1 \iff node j is on boundary
    std::unordered_map<int, std::vector<int>> node_to_cells_;   // for each node, the ids of cells sharing it
    SVector<2> range_ {};                                       // mesh bounding box (min and max coordinates)
    int n_nodes_ = 0, n_cells_ = 0;
    mutable std::optional<LocationPolicy> location_policy_ {};
};

}   // namespace fdapde

#endif   // __LINEAR_NETWORK_H__
