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
#include <unsupported/Eigen/SparseExtra>

#include "../utils/symbols.h"
#include "triangulation.h"

namespace fdapde {
namespace core {

// sort the range in clockwise order around their geometrical center
template <typename T> struct clockwise_order {
   private:
    T c_ {};
   public:
    clockwise_order(const T& c) : c_(c) { }
    bool operator()(const T& a, const T& b) {
        if (a[0] - c_[0] >= 0 && b[0] - c_[0] < 0) return true;
        if (b[0] - c_[0] >= 0 && a[0] - c_[0] < 0) return false;
        if (a[0] - c_[0] == 0 && b[0] - c_[0] == 0) {
            return (a[1] - c_[1] >= 0 || b[1] - c_[1] >= 0) ? a[1] > b[1] : b[1] > a[1];
        }
        // check sign of the cross product of vectors CA and CB
        double aXb_sign = (a[0] - c_[0]) * (b[1] - c_[1]) - (b[0] - c_[0]) * (a[1] - c_[1]);
        if (aXb_sign < 0) return true;
        if (aXb_sign > 0) return false;
        // points a and b are on the same line from the center, sort wrt distance from the center
        return (a - c_).squaredNorm() > (b - c_).squaredNorm();
    }
};

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
    int n_sites() const { return mesh_->n_nodes(); }
    const DMatrix<double>& sites() const { return mesh_->nodes(); }
    int n_vertices() const { return nodes_.rows(); }
    SVector<embed_dim> vertex(int id) const { return nodes_.row(id); }
    SVector<embed_dim> site(int id) const { return mesh_->node(id); }
    const BinaryVector<fdapde::Dynamic>& boundary_nodes() const { return nodes_markers_; }
    const Triangulation& dual() const { return *mesh_; }
  // n_vertices, n_faces

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

    class Cell {
       private:
        const Voronoi* v_;
        int id_ = 0;
       public:
        Cell() = default;
        Cell(int id, const Voronoi* v) : v_(v), id_(id) { }
        DMatrix<int> edges() const {
            DMatrix<int> result;
            int n_edges = v_->cells_.at(id_).size();
            result.resize(n_edges, local_dim);
            for (int j = 0; j < n_edges; ++j) {
	      for (int k = 0; k < local_dim; ++k) result(j, k) = v_->cells_.at(id_)[(j + k) % n_edges];
            }
            return result;
        }
        double measure() const {
            double area = 0;
            int n_edges = v_->cells_.at(id_).size();
            for (int j = 0; j < n_edges; ++j) {
                // compute doubled area of triangle connecting the j-th edge and the center (use cross product)
                SVector<embed_dim> x = v_->vertex(v_->cells_.at(id_)[j]);
                SVector<embed_dim> y = v_->vertex(v_->cells_.at(id_)[(j + 1) % n_edges]);
                area += x[0] * y[1] - x[1] * y[0];
            }
            return 0.5 * std::abs(area);
        }
    };
    Cell cell(int id) const { return Cell(id, this); }

    // iterator over voronoi cells
    struct cell_iterator {
       private:
        int index_;   // current element
        const Voronoi* voronoi_;
        Cell f_;
       public:
        using value_type        = Cell;
        using pointer           = const Cell*;
        using reference         = const Cell&;
        using size_type         = std::size_t;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        cell_iterator(int index, const Voronoi* voronoi) : index_(index), voronoi_(voronoi) {
            if (index_ < voronoi_->n_sites()) f_ = voronoi_->cell(index_);
        }
        reference operator*() const { return f_; }
        pointer operator->() const { return &f_; }
      
        cell_iterator& operator++() {
            ++index_;
            if (index_ < voronoi_->n_sites()) f_ = voronoi_->cell(index_);
            return *this;
        }
        cell_iterator operator++(int) {
            cell_iterator tmp(index_, this);
            ++(*this);
            return tmp;
        }
        friend bool operator!=(const cell_iterator& lhs, const cell_iterator& rhs) { return lhs.index_ != rhs.index_; }
        friend bool operator==(const cell_iterator& lhs, const cell_iterator& rhs) { return lhs.index_ == rhs.index_; }
    };
    cell_iterator cells_begin() const { return cell_iterator(0, this); }
    cell_iterator cells_end() const { return cell_iterator(n_sites(), this); }

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
                value.begin(), value.end(), mean,
                [&](const auto& c, int a) { return c + nodes_.row(a).transpose(); }) /
	      value.size());
            std::sort(value.begin(), value.end(), [&](int i, int j) { return compare(nodes_.row(i), nodes_.row(j)); });
        }
    }

    // perform point location for query point p
    // int locate(const SVector<embed_dim>& p) const {
    //     int dual_p = mesh_->locate(p);
    //     if (dual_p == -1) return dual_p;
    //     // find delanuay cell nearest vertex to i-th location
    //     typename Triangulation::FaceType f = mesh_->face(dual_p);
    //     SMatrix<1, Triangulation::n_vertices_per_face> dist = (f.vertices().colwise() - p).colwise().squaredNorm();
    //     int min_index;
    //     dist.minCoeff(&min_index);
    //     return f.node_ids()[min_index];
    // }
    // perform point location for a set of points
    DVector<int> locate(const DMatrix<double>& locs) const {
        fdapde_assert(locs.cols() == Triangulation::embed_dim);
        // find cells in the triangulation containing points
        DVector<int> dual_locs = mesh_->locate(locs);
        for (int i = 0; i < locs.rows(); ++i) {
            if (dual_locs[i] == -1) continue;   // location outside domain
            // find delanuay cell nearest vertex to i-th location
            typename Triangulation::FaceType f = mesh_->face(dual_locs[i]);
            SMatrix<1, Triangulation::n_vertices_per_face> dist =
              (f.vertices().colwise() - locs.row(i).transpose()).colwise().squaredNorm();
            int min_index;
            dist.minCoeff(&min_index);
            dual_locs[i] = f.node_ids()[min_index];
        }
        return dual_locs;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __VORONOI_H__
