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

#ifndef __FDAPDE_DCEL_H__
#define __FDAPDE_DCEL_H__

#include <map>
#include <set>
#include <stdexcept>

#include "header_check.h"

namespace fdapde {

namespace internals {
namespace delaunay_2d {
class dcel_inserter;
}
}   // namespace internals

/**
 * @brief index-based double connected edge list for planar triangular meshes
 *
 * Topology and conflict-graph usage are adapted from Luca D'Alessandro and Francesca Zambetti,
 * francesca1606/Luca-Francesca stable@fbd20e1. Storage uses integer handles instead of self-referential pointers, so
 * copies and moves are safe and topology traversal remains constant time.
 */
template <int LocalDim, int EmbedDim> class DCEL {
    static_assert(LocalDim == 2 && EmbedDim == 2, "DCEL currently supports planar triangular meshes only");
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;
    static constexpr int invalid_id = -1;
    using edge_t = std::array<int, 2>;
    using coords_t = Matrix<double, 1, embed_dim>;

    class node_t {
       public:
        int id() const { return id_; }
        int halfedge() const { return halfedge_; }
        bool on_boundary() const { return boundary_; }
        const coords_t& coords() const { return coords_; }
       private:
        friend class DCEL;
        friend class internals::delaunay_2d::dcel_inserter;
        int id_ = invalid_id;
        int halfedge_ = invalid_id;
        bool boundary_ = false;
        coords_t coords_ {};
    };

    class halfedge_t {
       public:
        int id() const { return id_; }
        int origin() const { return origin_; }
        int twin() const { return twin_; }
        int previous() const { return previous_; }
        int next() const { return next_; }
        int cell() const { return cell_; }
        bool on_boundary() const { return boundary_; }
        bool is_segment() const { return segment_; }
       private:
        friend class DCEL;
        friend class internals::delaunay_2d::dcel_inserter;
        int id_ = invalid_id;
        int origin_ = invalid_id;
        int twin_ = invalid_id;
        int previous_ = invalid_id;
        int next_ = invalid_id;
        int cell_ = invalid_id;
        bool boundary_ = false;
        bool segment_ = false;
        bool active_ = true;
    };

    class cell_t {
       public:
        int id() const { return id_; }
        int halfedge() const { return halfedge_; }
       private:
        friend class DCEL;
        friend class internals::delaunay_2d::dcel_inserter;
        int id_ = invalid_id;
        int halfedge_ = invalid_id;
        bool active_ = true;
    };

    using const_node_iterator = typename std::vector<node_t>::const_iterator;
    using const_halfedge_iterator = typename std::vector<halfedge_t>::const_iterator;
    using const_cell_iterator = typename std::vector<cell_t>::const_iterator;

    DCEL() = default;

    static DCEL from_triangles(
      const Matrix<double, Dynamic, Dynamic>& nodes, const Matrix<int, Dynamic, Dynamic>& cells,
      const std::set<edge_t>& segments = {}) {
        if (nodes.rows() == 0 || nodes.cols() != embed_dim) {
            throw std::invalid_argument("DCEL nodes must be a nonempty two-column matrix.");
        }
        if (cells.rows() == 0 || cells.cols() != 3) {
            throw std::invalid_argument("DCEL cells must be a nonempty three-column matrix.");
        }

        DCEL result;
        result.nodes_.resize(nodes.rows());
        for (int i = 0; i < nodes.rows(); ++i) {
            if (!std::isfinite(nodes(i, 0)) || !std::isfinite(nodes(i, 1))) {
                throw std::invalid_argument("DCEL nodes must contain only finite coordinates.");
            }
            result.nodes_[i].id_ = i;
            result.nodes_[i].coords_ = nodes.row(i);
        }

        result.cells_.resize(cells.rows());
        result.halfedges_.reserve(3 * cells.rows() + nodes.rows());
        std::map<edge_t, std::vector<int>> edge_groups;
        for (int i = 0; i < cells.rows(); ++i) {
            std::array<int, 3> cell = {cells(i, 0), cells(i, 1), cells(i, 2)};
            for (int node : cell) {
                if (node < 0 || node >= nodes.rows()) {
                    throw std::invalid_argument("DCEL cell contains an out-of-range node index.");
                }
            }
            if (cell[0] == cell[1] || cell[1] == cell[2] || cell[2] == cell[0]) {
                throw std::invalid_argument("DCEL cell contains a repeated node.");
            }
            const double orientation = orientation_(nodes, cell);
            if (orientation == 0.0) { throw std::invalid_argument("DCEL cell has zero signed area."); }
            if (orientation < 0.0) std::swap(cell[1], cell[2]);

            result.cells_[i].id_ = i;
            const int first = result.halfedges_.size();
            result.cells_[i].halfedge_ = first;
            for (int j = 0; j < 3; ++j) {
                halfedge_t halfedge;
                halfedge.id_ = first + j;
                halfedge.origin_ = cell[j];
                halfedge.previous_ = first + (j + 2) % 3;
                halfedge.next_ = first + (j + 1) % 3;
                halfedge.cell_ = i;
                const edge_t edge = normalize_edge_(cell[j], cell[(j + 1) % 3]);
                halfedge.segment_ = segments.contains(edge);
                result.halfedges_.push_back(halfedge);
                edge_groups[edge].push_back(halfedge.id_);
                if (result.nodes_[cell[j]].halfedge_ == invalid_id) result.nodes_[cell[j]].halfedge_ = halfedge.id_;
            }
        }

        std::vector<int> external;
        for (const auto& [edge, group] : edge_groups) {
            if (group.size() > 2) { throw std::invalid_argument("DCEL contains a non-manifold edge."); }
            if (group.size() == 2) {
                halfedge_t& first = result.halfedges_[group[0]];
                halfedge_t& second = result.halfedges_[group[1]];
                if (first.origin_ == second.origin_) {
                    throw std::invalid_argument("DCEL adjacent cells have inconsistent orientation.");
                }
                first.twin_ = second.id_;
                second.twin_ = first.id_;
                continue;
            }

            halfedge_t& interior = result.halfedges_[group.front()];
            halfedge_t outside;
            outside.id_ = result.halfedges_.size();
            outside.origin_ = result.halfedges_[interior.next_].origin_;
            outside.twin_ = interior.id_;
            outside.cell_ = invalid_id;
            outside.boundary_ = true;
            outside.segment_ = true;
            interior.twin_ = outside.id_;
            interior.boundary_ = true;
            interior.segment_ = true;
            result.nodes_[interior.origin_].boundary_ = true;
            result.nodes_[outside.origin_].boundary_ = true;
            result.halfedges_.push_back(outside);
            external.push_back(outside.id_);
        }

        std::map<int, int> external_from;
        for (int id : external) {
            const int origin = result.halfedges_[id].origin_;
            if (!external_from.emplace(origin, id).second) {
                throw std::invalid_argument("DCEL boundary is non-manifold at a node.");
            }
            if (result.nodes_[origin].halfedge_ == invalid_id) result.nodes_[origin].halfedge_ = id;
        }
        for (int id : external) {
            halfedge_t& halfedge = result.halfedges_[id];
            const int destination = result.halfedges_[halfedge.twin_].origin_;
            const auto next = external_from.find(destination);
            if (next == external_from.end()) { throw std::invalid_argument("DCEL boundary is not a closed cycle."); }
            halfedge.next_ = next->second;
            result.halfedges_[next->second].previous_ = id;
        }
        return result;
    }

    static DCEL from_triangulation(const Triangulation<2, embed_dim>& triangulation) {
        std::set<edge_t> boundary;
        const auto edges = triangulation.edges();
        for (int i = 0; i < triangulation.n_edges(); ++i) {
            if (triangulation.is_edge_on_boundary(i)) boundary.insert(normalize_edge_(edges(i, 0), edges(i, 1)));
        }
        return from_triangles(triangulation.nodes(), triangulation.cells(), boundary);
    }

    Triangulation<2, embed_dim> triangulation() const {
        Matrix<int, Dynamic, Dynamic> boundary = Matrix<int, Dynamic, Dynamic>::Zero(nodes_.size(), 1);
        for (const node_t& node : nodes_) boundary(node.id_, 0) = node.boundary_ ? 1 : 0;
        return Triangulation<2, embed_dim>(nodes(), cells(), boundary);
    }

    Matrix<double, Dynamic, Dynamic> nodes() const {
        Matrix<double, Dynamic, Dynamic> result(nodes_.size(), embed_dim);
        for (const node_t& node : nodes_) result.row(node.id_) = node.coords_;
        return result;
    }

    Matrix<int, Dynamic, Dynamic> cells() const {
        Matrix<int, Dynamic, Dynamic> result(cells_.size(), 3);
        for (const cell_t& cell : cells_) {
            const auto nodes = cell_nodes(cell.id_);
            result.row(cell.id_) = {nodes[0], nodes[1], nodes[2]};
        }
        return result;
    }

    const node_t& node(int id) const { return nodes_.at(id); }
    const halfedge_t& halfedge(int id) const { return halfedges_.at(id); }
    const cell_t& cell(int id) const { return cells_.at(id); }
    int destination(int halfedge) const { return halfedges_.at(halfedges_.at(halfedge).twin_).origin_; }
    int adjacent_cell(int halfedge) const { return halfedges_.at(halfedges_.at(halfedge).twin_).cell_; }

    std::array<int, 3> cell_halfedges(int cell) const {
        const int first = cells_.at(cell).halfedge_;
        const int second = halfedges_.at(first).next_;
        const int third = halfedges_.at(second).next_;
        if (halfedges_.at(third).next_ != first) { throw std::runtime_error("DCEL cell is not triangular."); }
        return {first, second, third};
    }

    std::array<int, 3> cell_nodes(int cell) const {
        const auto halfedges = cell_halfedges(cell);
        return {halfedges_[halfedges[0]].origin_, halfedges_[halfedges[1]].origin_, halfedges_[halfedges[2]].origin_};
    }

    int n_nodes() const { return nodes_.size(); }
    int n_halfedges() const { return halfedges_.size(); }
    int n_edges() const { return halfedges_.size() / 2; }
    int n_cells() const { return cells_.size(); }
    int n_boundary_edges() const {
        return std::count_if(halfedges_.begin(), halfedges_.end(), [](const halfedge_t& halfedge) {
            return halfedge.boundary_ && halfedge.id_ < halfedge.twin_;
        });
    }

    const_node_iterator nodes_begin() const { return nodes_.begin(); }
    const_node_iterator nodes_end() const { return nodes_.end(); }
    const_halfedge_iterator halfedges_begin() const { return halfedges_.begin(); }
    const_halfedge_iterator halfedges_end() const { return halfedges_.end(); }
    const_cell_iterator cells_begin() const { return cells_.begin(); }
    const_cell_iterator cells_end() const { return cells_.end(); }
   private:
    friend class internals::delaunay_2d::dcel_inserter;

    static edge_t normalize_edge_(int first, int second) {
        return first < second ? edge_t {first, second} : edge_t {second, first};
    }

    static double orientation_(const Matrix<double, Dynamic, Dynamic>& nodes, const std::array<int, 3>& cell) {
        double scale = 0.0;
        for (int node : cell) scale = std::max({scale, std::abs(nodes(node, 0)), std::abs(nodes(node, 1))});
        const int shift = scale == 0.0 ? 0 : -std::ilogb(scale);
        const auto coordinate = [&](int node, int dimension) { return std::scalbn(nodes(node, dimension), shift); };
        return (coordinate(cell[1], 0) - coordinate(cell[0], 0)) * (coordinate(cell[2], 1) - coordinate(cell[0], 1)) -
               (coordinate(cell[1], 1) - coordinate(cell[0], 1)) * (coordinate(cell[2], 0) - coordinate(cell[0], 0));
    }

    std::vector<node_t> nodes_;
    std::vector<halfedge_t> halfedges_;
    std::vector<cell_t> cells_;
};

template <int EmbedDim> DCEL<2, EmbedDim> to_dcel(const Triangulation<2, EmbedDim>& triangulation) {
    return DCEL<2, EmbedDim>::from_triangulation(triangulation);
}

template <int EmbedDim> Triangulation<2, EmbedDim> to_triangulation(const DCEL<2, EmbedDim>& dcel) {
    return dcel.triangulation();
}

}   // namespace fdapde

#endif   // __FDAPDE_DCEL_H__
