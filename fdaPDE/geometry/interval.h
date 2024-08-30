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

#ifndef __INTERVAL_H__
#define __INTERVAL_H__

#include "segment.h"
#include "../linear_algebra/binary_matrix.h"

namespace fdapde {
  
// template specialization for 1D meshes (bounded intervals)
template <int M, int N> class Triangulation;
template <> class Triangulation<1, 1> {
   public:
    static constexpr int local_dim = 1;
    static constexpr int embed_dim = 1;
    static constexpr int n_nodes_per_cell = 2;
    static constexpr int n_neighbors_per_cell = 2;
    static constexpr bool is_manifold = false;
    using CellType = Segment<Triangulation<1, 1>>;
    using VertexType = SVector<1>;
  
    Triangulation() = default;
    Triangulation(const DVector<double>& nodes) : nodes_(nodes) {
        // store number of nodes and elements
        n_nodes_ = nodes_.rows();
        n_cells_ = n_nodes_ - 1;
        // compute mesh limits
        range_[0] = nodes_[0];
        range_[1] = nodes_[n_nodes_ - 1];
        // build elements and neighboring structure
        cells_.resize(n_cells_, 2);
        for (int i = 0; i < n_nodes_ - 1; ++i) {
            cells_(i, 0) = i;
            cells_(i, 1) = i + 1;
        }
        neighbors_ = DMatrix<int>::Constant(n_cells_, n_neighbors_per_cell, -1);
        neighbors_(0, 1) = 1;
        for (int i = 1; i < n_cells_ - 1; ++i) {
            neighbors_(i, 0) = i - 1;
            neighbors_(i, 1) = i + 1;
        }
        neighbors_(n_cells_ - 1, 0) = n_cells_ - 2;
	// set first and last nodes as boundary nodes
	nodes_markers_.resize(n_nodes_);
	nodes_markers_.set(0);
	nodes_markers_.set(n_nodes_ - 1);
    };
    // construct from interval's bounds [a, b] and the number of subintervals n into which split [a, b]
    Triangulation(double a, double b, int n) : Triangulation(DVector<double>::LinSpaced(n + 1, a, b)) { }

    // getters
    CellType cell(int id) const { return CellType(id, this); }
    VertexType node(int id) const { return SVector<1>(nodes_[id]); }
    bool is_node_on_boundary(int id) const { return (id == 0 || id == (n_nodes_ - 1)); }
    const DVector<double>& nodes() const { return nodes_; }
    const DMatrix<int, Eigen::RowMajor>& cells() const { return cells_; }
    const DMatrix<int, Eigen::RowMajor>& neighbors() const { return neighbors_; }
    const BinaryVector<fdapde::Dynamic>& boundary() const { return nodes_markers_; }
    int n_cells() const { return n_cells_; }
    int n_nodes() const { return n_nodes_; }
    SVector<2> range() const { return range_; }

    // iterators support
    class cell_iterator : public internals::index_iterator<cell_iterator, CellType> {
        using Base = internals::index_iterator<cell_iterator, CellType>;
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

    // localize element containing point using a O(log(n)) time-complexity binary search strategy
    DVector<int> locate(const DVector<double>& points) const {
        // allocate space
        DVector<int> result;
        result.resize(points.rows());
        // start search
        for (int i = 0; i < points.rows(); ++i) {
            // check if point is inside
            if (points[i] < range_[0] || points[i] > range_[1]) {
                result[i] = -1;
            } else {
                // binary search strategy
                int h_min = 0, h_max = n_nodes_;
                while (true) {
                    int j = h_min + std::floor((h_max - h_min) / 2);
                    if (points[i] >= nodes_[j] && points[i] < nodes_[j + 1]) {
                        result[i] = j;
                        break;
                    } else {
                        if (points[i] < nodes_[j]) {
                            h_max = j;
                        } else {
                            h_min = j;
                        }
                    }
                }
            }
        }
        return result;
    }
   protected:
    DVector<double> nodes_;                            // physical coordinates of mesh's vertices
    DMatrix<int, Eigen::RowMajor> cells_ {};           // nodes (as row indexes in nodes_ matrix) composing each cell
    DMatrix<int, Eigen::RowMajor> neighbors_ {};       // ids of faces adjacent to a given face (-1 if no adjacent face)
    BinaryVector<fdapde::Dynamic> nodes_markers_ {};   // j-th element is 1 \iff node j is on boundary
    SVector<2> range_ {};                              // mesh bounding box (min and max coordinates)
    int n_nodes_ = 0, n_cells_ = 0;
};

}   // namespace fdapde

#endif   // __INTERVAL_H__
