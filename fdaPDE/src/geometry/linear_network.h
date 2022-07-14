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

#ifndef __FDAPDE_LINEAR_NETWORK_H__
#define __FDAPDE_LINEAR_NETWORK_H__

#include "header_check.h"

namespace fdapde {
  
// template specialization for 1D meshes (bounded intervals)
template <int LocalDim, int EmbedDim> class Triangulation;
template <> class Triangulation<1, 2> : public TriangulationBase<1, 2, Triangulation<1, 2>> {
   public:
    using Base = TriangulationBase<1, 2, Triangulation<1, 2>>;
    static constexpr int local_dim = 1;
    static constexpr int embed_dim = 2;
    static constexpr int n_nodes_per_cell = 2;
    static constexpr int n_neighbors_per_cell = Dynamic;
    static constexpr bool is_manifold = true;
    using LocationPolicy = TreeSearch<Triangulation<1, 2>>;

    struct CellType : public Segment<Triangulation<1, 2>> {
       private:
        using Base = Segment<Triangulation<1, 2>>;
        using Base::id_;
        using Base::mesh_;
       public:
        CellType() = default;
        CellType(int id, const Triangulation* mesh) : Segment<Triangulation<1, 2>>(id, mesh) { }

        std::vector<int> neighbors() const {
            const auto& v1 = mesh_->node_to_cells_.at(mesh_->cells()(id_, 0));
            const auto& v2 = mesh_->node_to_cells_.at(mesh_->cells()(id_, 1));
	    std::vector<int> result(v1.size() + v2.size());
            int i = 0;
            for (; i < v1.size(); ++i) result[i] = v1[i];
            for (; i < v1.size() + v2.size(); ++i) result[i] = v2[i];
            return result;
        }
    };

    Triangulation() = default;
    Triangulation(
      const Eigen::Matrix<double, Dynamic, Dynamic>& nodes, const Eigen::Matrix<int, Dynamic, Dynamic>& cells,
      const Eigen::Matrix<int, Dynamic, Dynamic>& boundary, int flags = 0) :
        Base(nodes, cells, boundary, flags) {
        // store number of nodes and number of elements
        n_nodes_ = nodes_.rows();
        n_cells_ = cells_.rows();
        if (flags_ & cache_cells) {   // populate cache if cell caching is active
            cell_cache_.reserve(n_cells_);
            for (int i = 0; i < n_cells_; ++i) { cell_cache_.emplace_back(i, this); }
        }
        // compute mesh limits
        bbox_.row(0) = nodes_.colwise().minCoeff();
        bbox_.row(1) = nodes_.colwise().maxCoeff();
        neighbors_.resize(n_cells_, n_cells_);
        // compute node to cells boundings
        for (int i = 0; i < n_cells_; ++i) {
            node_to_cells_[cells_(i, 0)].push_back(i);
            node_to_cells_[cells_(i, 1)].push_back(i);
        }
        // recover adjacency matrix
        Eigen::SparseMatrix<int> adjoint_neighbors;
        std::vector<Eigen::Triplet<int>> triplet_list;
        for (const auto& [node, edges] : node_to_cells_) {
            for (int i = 0; i < edges.size(); ++i) {
                for (int j = i + 1; j < edges.size(); ++j) { triplet_list.emplace_back(edges[j], edges[i], 1); }
            }
        }
        adjoint_neighbors.resize(n_cells_, n_cells_);
        adjoint_neighbors.setFromTriplets(triplet_list.begin(), triplet_list.end());
        neighbors_ = adjoint_neighbors.selfadjointView<Eigen::Lower>();   // symmetrize neighboring relation
    }
    Triangulation(
      const std::string& nodes, const std::string& cells, const std::string& boundary, bool header, bool index_col,
      int flags = 0) :
        Triangulation(
          read_table<double>(nodes, header, index_col).as_matrix(),
          read_table<int>(cells, header, index_col).as_matrix(),
          read_table<int>(boundary, header, index_col).as_matrix(), flags) { }

    // getters
    const Eigen::SparseMatrix<int>& neighbors() const { return neighbors_; }
    const CellType& cell(int id) const {
        if (Base::flags_ & cache_cells) {   // cell caching enabled
            return cell_cache_[id];
        } else {
            cell_ = CellType(id, this);
            return cell_;
        }
    }
    // boundary iterator
    using boundary_iterator = Base::boundary_node_iterator;
    BoundaryIterator<Triangulation<1, 2>> boundary_begin(int marker = BoundaryAll) const {
        return BoundaryIterator<Triangulation<1, 2>>(0, this, marker);
    }
    BoundaryIterator<Triangulation<1, 2>> boundary_end(int marker = BoundaryAll) const {
        return BoundaryIterator<Triangulation<1, 2>>(n_boundary_nodes(), this, marker);
    }
    std::pair<BoundaryIterator<Triangulation<1, 2>>, BoundaryIterator<Triangulation<1, 2>>>
    boundary(int marker = BoundaryAll) const {
        return std::make_pair(boundary_begin(marker), boundary_end(marker));
    }
    // set boundary markers
    template <typename Lambda>
    void mark_boundary(int marker, Lambda&& lambda)
        requires(requires(Lambda lambda, typename Base::NodeType e) {
            { lambda(e) } -> std::same_as<bool>;
        }) {
        fdapde_assert(marker >= 0);
        nodes_markers_.resize(n_nodes_);
        for (boundary_node_iterator it = boundary_nodes_begin(); it != boundary_nodes_end(); ++it) {
            nodes_markers_[it->id()] = lambda(*it) ? marker : Unmarked;
        }
    }
    template <int Rows, typename XprType> void mark_boundary(const BinMtxBase<Rows, 1, XprType>& mask) {
        fdapde_assert(mask.rows() == n_nodes_);
        nodes_markers_.resize(n_nodes_);
        for (boundary_node_iterator it = boundary_nodes_begin(); it != boundary_nodes_end(); ++it) {
            nodes_markers_[it->id()] = mask[it->id()] ? 1 : 0;
        }
    }
    template <typename Iterator> void mark_boundary(Iterator first, Iterator last) {
        fdapde_static_assert(
          std::is_convertible_v<typename Iterator::value_type FDAPDE_COMMA int>, INVALID_ITERATOR_RANGE);
        int n_markers = std::distance(first, last);
	bool all_markers_positive = std::all_of(first, last, [](auto marker) { return marker >= 0; });
        fdapde_assert(n_markers == n_nodes() && all_markers_positive);
        nodes_markers_.resize(n_nodes_, Unmarked);
        for (int i = 0; i < n_nodes_; ++i) { nodes_markers_[i] = *(first + i); }
    }
    // marks all boundary edges
    void mark_boundary(int marker) {
        fdapde_assert(marker >= 0);
        nodes_markers_.resize(n_nodes_, Unmarked);
        std::for_each(nodes_markers_.begin(), nodes_markers_.end(), [marker](int& marker_) { marker_ = marker; });
    }
    void clear_boundary_markers() {
        std::for_each(nodes_markers_.begin(), nodes_markers_.end(), [](int& marker) { marker = Unmarked; });
    }
    // point location
    template <int Rows, int Cols>
    std::conditional_t<Rows == Dynamic || Cols == Dynamic, Eigen::Matrix<int, Dynamic, 1>, int>
    locate(const Eigen::Matrix<double, Rows, Cols>& p) const {
        fdapde_static_assert(
          (Cols == 1 && Rows == embed_dim) || (Cols == Dynamic && Rows == Dynamic),
          YOU_PASSED_A_MATRIX_OF_POINTS_TO_LOCATE_OF_WRONG_DIMENSIONS);
        if (!location_policy_.has_value()) { location_policy_ = LocationPolicy(this); }
        return location_policy_->locate(p);
    }
    template <typename Derived> Eigen::Matrix<int, Dynamic, 1> locate(const Eigen::Map<Derived>& p) const {
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->locate(p);
    }
    // the set of cells which have node id as vertex
    std::vector<int> node_patch(int id) const { return node_to_cells_.at(id); }
   protected:
    Eigen::SparseMatrix<int> neighbors_ {};   // ids of faces adjacent to a given face (-1 if no adjacent face)
    std::unordered_map<int, std::vector<int>> node_to_cells_ {};   // for each node, the ids of cells sharing it
    mutable std::optional<LocationPolicy> location_policy_ {};
    // cell caching
    std::vector<CellType> cell_cache_;
    mutable CellType cell_;   // used in case cell caching is off
};

}   // namespace fdapde

#endif   // __FDAPDE_LINEAR_NETWORK_H__
