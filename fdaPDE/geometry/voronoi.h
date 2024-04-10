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

#include <unordered_map>
#include <unordered_set>

#include "../utils/symbols.h"
#include "utils.h"

namespace fdapde {
namespace core {

// adaptor adapting a (Delanoy) triangulation to its dual Vornoi diagram
template <typename Triangulation> class Voronoi {
   private:
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int n_vertices_per_face = Triangulation::n_vertices_per_face;

    const Triangulation* mesh_;
    DMatrix<double> nodes_;                             // voronoi vertices
    BinaryVector<fdapde::Dynamic> nodes_markers_;       // i-th element true if i-th vertex is on boundary
    std::unordered_map<int, std::vector<int>> cells_;   // for each cell id, the ids of the vertices composing it
   public:
    Voronoi() = default;
    Voronoi(const Triangulation& mesh) : mesh_(&mesh) {   // constructs voronoi diagram from Delanoy triangulation
        int n_delaunay_faces = mesh_->n_faces();
        int n_delaunay_boundary_edges = mesh_->n_boundary_edges();
        nodes_.resize(n_delaunay_faces + n_delaunay_boundary_edges + mesh_->n_boundary_nodes(), embed_dim);
        nodes_markers_.resize(nodes_.rows());
        int k = n_delaunay_faces;
        for (typename Triangulation::face_iterator it = mesh_->faces_begin(); it != mesh_->faces_end(); ++it) {
            nodes_.row(it->id()) = it->circumcenter();
            for (int v : it->node_ids()) { cells_[v].push_back(it->id()); }
            if (it->on_boundary()) {
                for (typename Triangulation::FaceType::face_iterator jt = it->faces_begin(); jt != it->faces_end();
                     ++jt) {
                    if (jt->on_boundary()) {
                        nodes_.row(k) = jt->supporting_plane().project(nodes_.row(it->id()));
                        nodes_markers_.set(k);
                        for (int v : jt->node_ids()) { cells_[v].push_back(k); }
                        k++;
                    }
                }
            }
        }
        // augment node set with boundary vertices, sort each cell clockwise (around its mean point)
        for (auto& [key, value] : cells_) {
            if (mesh_->is_node_on_boundary(key)) {
                nodes_.row(k) = mesh_->node(key);
                nodes_markers_.set(k);
                value.push_back(k);
                k++;
            }
            SVector<embed_dim> mean = SVector<embed_dim>::Zero();
            auto compare = clockwise_order<SVector<embed_dim>>(
              std::accumulate(
                value.begin(), value.end(), mean, [&](const auto& c, int a) { return c + nodes_.row(a).transpose(); }) /
              value.size());
            std::sort(value.begin(), value.end(), [&](int i, int j) { return compare(nodes_.row(i), nodes_.row(j)); });
        }
    }

    // cell data structure
    class VoronoiCell {
       private:
        const Voronoi* v_;
        int id_ = 0;
        int n_edges_ = 0;
       public:
        VoronoiCell() = default;
      VoronoiCell(int id, const Voronoi* v) : v_(v), id_(id), n_edges_(v_->cells_.at(id_).size()) { }
        // matrix of edge identifiers
        DMatrix<int> edges() const {
            DMatrix<int> result;
            result.resize(n_edges_, local_dim);
            for (int j = 0; j < n_edges_; ++j) {
                for (int k = 0; k < local_dim; ++k) result(j, k) = v_->cells_.at(id_)[(j + k) % n_edges_];
            }
            return result;
        }
        double measure() const {
            double area = 0;
            for (int j = 0; j < n_edges_; ++j) {
                // compute doubled area of triangle connecting the j-th edge and the center (use cross product)
                SVector<embed_dim> x = v_->vertex(v_->cells_.at(id_)[j]);
                SVector<embed_dim> y = v_->vertex(v_->cells_.at(id_)[(j + 1) % n_edges_]);
                area += x[0] * y[1] - x[1] * y[0];
            }
            return 0.5 * std::abs(area);
        }
        typename Triangulation::EdgeType edge(int i) const {
            fdapde_assert(i < n_edges_);
            SMatrix<embed_dim, 2> coords;
            for (int k = 0; k < embed_dim; ++k) { coords.col(k) = v_->vertex(v_->cells_.at(id_)[(i + k) % n_edges_]); }
            return typename Triangulation::EdgeType(coords);
        }
        bool on_boundary() const {
            for (int j = 0; j < n_edges_; ++j) {
                bool boundary = true;
                for (int k = 0; k < local_dim; ++k)
                    boundary &= v_->nodes_markers_[v_->cells_.at(id_)[(j + k) % n_edges_]];
                if (boundary == true) return true;
            }
            return false;   // no edge on boundary
        };
    };

    // getters
    const DMatrix<double>& sites() const { return mesh_->nodes(); }
    SVector<embed_dim> vertex(int id) const { return nodes_.row(id); }
    SVector<embed_dim> site(int id) const { return mesh_->node(id); }
    const BinaryVector<fdapde::Dynamic>& boundary_vertices() const { return nodes_markers_; }
    const DMatrix<double>& vertices() const { return nodes_; }
    const Triangulation& dual() const { return *mesh_; }
    int n_vertices() const { return nodes_.rows(); }
    int n_cells() const { return mesh_->n_nodes(); }
    // compute matrix of edges
    DMatrix<int> edges() const {
        std::unordered_set<std::array<int, local_dim>, std_array_hash<int, local_dim>> visited;
        std::array<int, local_dim> edge;
        for (const auto& [key, value] : cells_) {
            int n_edges = value.size();
            for (int j = 0; j < n_edges; ++j) {
                for (int k = 0; k < local_dim; ++k) { edge[k] = value[(j + k) % n_edges]; }
                std::sort(edge.begin(), edge.end());
                if (visited.find(edge) == visited.end()) { visited.insert(edge); }
            }
        }
        DMatrix<int> result;
        result.resize(visited.size(), local_dim);
        int i = 0;
        for (const auto& e : visited) {
            for (int k = 0; k < local_dim; ++k) result(i, k) = e[k];
            i++;
        }
        return result;
    }
    VoronoiCell cell(int id) const { return VoronoiCell(id, this); }

    // iterators
    class cell_iterator : public index_based_iterator<cell_iterator, VoronoiCell> {
        using Base = index_based_iterator<cell_iterator, VoronoiCell>;
        using Base::index_;
        const Voronoi* voronoi_;
       public:
        cell_iterator(int index, const Voronoi* voronoi) : Base(index), voronoi_(voronoi) {
            if (index_ < voronoi_->n_cells()) this->val_ = voronoi_->cell(index_);
        }
        cell_iterator& operator++() {
            ++index_;
            if (index_ < voronoi_->n_cells()) this->val_ = voronoi_->cell(index_);
            return *this;
        }
        cell_iterator& operator--() {
            --index_;
            if (index_ >= 0) this->val_ = voronoi_->cell(index_);
            return *this;
        }
    };
    cell_iterator cells_begin() const { return cell_iterator(0, this); }
    cell_iterator cells_end() const { return cell_iterator(n_cells(), this); }

    // perform point location for set of points p_1, p_2, \ldots, p_n
    DVector<int> locate(const DMatrix<double>& locs) const {
        fdapde_assert(locs.cols() == Triangulation::embed_dim);
        // find delanuay cells containing locs
        DVector<int> dual_locs = mesh_->locate(locs);
        for (int i = 0; i < locs.rows(); ++i) {
            if (dual_locs[i] == -1) continue;   // location outside domain
            // find nearest cell to i-th location
            typename Triangulation::FaceType f = mesh_->face(dual_locs[i]);
            SMatrix<1, Triangulation::n_vertices_per_face> dist =
              (f.vertices().colwise() - locs.row(i).transpose()).colwise().squaredNorm();
            int min_index;
            dist.minCoeff(&min_index);
            dual_locs[i] = f.node_ids()[min_index];
        }
        return dual_locs;
    }

    using CellType = VoronoiCell;
};

}   // namespace core
}   // namespace fdapde

#endif   // __VORONOI_H__
