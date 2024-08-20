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

#ifndef __DOF_HANDLER_H__
#define __DOF_HANDLER_H__

#include "../fields/polynomial.h"
#include "../geometry/triangulation.h"
#include "../utils/constexpr.h"
#include "../utils/symbols.h"
#include "dof_tetrahedron.h"
#include "dof_triangle.h"

namespace fdapde {

enum Boundary : short { All = -1, Dirichlet = 1, Neumann = 2, Robin = 3 };    // value 0 is for NOT-boundary nodes
  
template <int LocalDim, int EmbedDim> class DofHandler;
template <int LocalDim, int EmbedDim, typename Derived> class DofHandlerBase {
   public:
    using TriangulationType = Triangulation<LocalDim, EmbedDim>;
    using CellType = std::conditional_t<LocalDim == 2, DofTriangle<Derived>, DofTetrahedron<Derived>>;
    using MarkerType = short;
    static constexpr int n_nodes_per_cell = TriangulationType::n_nodes_per_cell;
    static constexpr int local_dim = TriangulationType::local_dim;
    static constexpr int embed_dim = TriangulationType::embed_dim;
    DofHandlerBase() = default;
    DofHandlerBase(const TriangulationType& triangulation) : triangulation_(&triangulation) { }
    // getters
    CellType cell(int id) const { return CellType(id, static_cast<const Derived*>(this)); }
    const DMatrix<int, Eigen::RowMajor>& dofs() const { return dofs_; }
    int n_dofs() const { return n_dofs_; }
    bool is_dof_on_boundary(int i) const { return boundary_dofs_[i]; }
    MarkerType dof_marker(int i) const { return dofs_markers_[i]; }
    const TriangulationType* triangulation() const { return triangulation_; }
    int n_boundary_dofs() const { return boundary_dofs_.count(); }
    int n_boundary_dofs(MarkerType marker) const {
        int i = 0, sum = 0;
        for (auto dof_marker : dofs_markers_) { sum += (dof_marker == marker && boundary_dofs_[i++]) ? 1 : 0; }
        return sum;
    }
    std::vector<int> filter_dofs_by_marker(MarkerType marker) const {
        std::vector<int> result;
        for (int i = 0; i < n_dofs_; ++i) {
            if (dofs_markers_[i] == marker) result.push_back(i);
        }
        return result;
    }
    DVector<int> active_dofs(int cell_id) const { return dofs_.row(cell_id); }
    bool has_markers() const { return is_empty(dofs_markers_); }
    operator bool() const { return n_dofs_ != 0; }

    // iterates over geometric cells coupled with dofs informations
    class cell_iterator : public index_based_iterator<cell_iterator, CellType> {
        using Base = index_based_iterator<cell_iterator, CellType>;
        using Base::index_;
        friend Base;
        const Derived* dof_handler_;
        cell_iterator& operator()(int i) {
            Base::val_ = dof_handler_->cell(i);
            return *this;
        }
       public:
        cell_iterator() = default;
        cell_iterator(int index, const Derived* dof_handler) :
	  Base(index, 0, dof_handler->triangulation()->n_cells()), dof_handler_(dof_handler) {
            if (index_ < dof_handler_->triangulation()->n_cells()) operator()(index_);
        }
    };
    cell_iterator cells_begin() const { return cell_iterator(0, static_cast<const Derived*>(this)); }
    cell_iterator cells_end() const {
        return cell_iterator(triangulation_->n_cells(), static_cast<const Derived*>(this));
    }

    class BoundaryDofType {
        int id_;
        const Derived* dof_handler_;
       public:
        BoundaryDofType() = default;
        BoundaryDofType(int id, const Derived* dof_handler) : id_(id), dof_handler_(dof_handler) { }
        int id() const { return id_; }
        MarkerType marker() const { return dof_handler_->dofs_markers_[id_]; }
        MarkerType& marker() { return dof_handler_->dofs_markers_[id_]; }
    };
    class boundary_dofs_iterator : public index_based_iterator<boundary_dofs_iterator, BoundaryDofType> {
        using Base = index_based_iterator<boundary_dofs_iterator, BoundaryDofType>;
        using Base::index_;
        const Derived* dof_handler_;
        MarkerType marker_;
       public:
        boundary_dofs_iterator(int index, const Derived* dof_handler, MarkerType marker) :
            Base(index, 0, dof_handler_->n_dofs()), dof_handler_(dof_handler), marker_(marker) {
            for (; index_ < dof_handler_->n_dofs() && !dof_handler_->boundary_dofs_[index_] &&
                   (!(dof_handler_->dof_marker(index_) == marker_) || marker_ == MarkerType(Boundary::All));
                 ++index_);
            this->val_ = BoundaryDofType(index_, dof_handler_);
        }
        boundary_dofs_iterator& operator++() {
            index_++;
            for (; index_ < dof_handler_->n_dofs() && !dof_handler_->boundary_dofs_[index_] &&
                   (!(dof_handler_->dof_marker(index_) == marker_) || marker_ == MarkerType(Boundary::All));
                 ++index_);
            this->val_ = BoundaryDofType(index_, dof_handler_);
            return *this;
        }
        boundary_dofs_iterator& operator--() {
            --index_;
            for (; index_ >= 0 && !dof_handler_->boundary_dofs_[index_] &&
                   (!(dof_handler_->dof_marker(index_) == marker_) || marker_ == MarkerType(Boundary::All));
                 --index_);
            this->val_ = BoundaryDofType(index_, dof_handler_);
            return *this;
        }
    };
    boundary_dofs_iterator boundary_dofs_begin(MarkerType marker) const {
        return boundary_dofs_iterator(0, static_cast<const Derived*>(this), marker);
    }
    boundary_dofs_iterator boundary_dofs_end(MarkerType marker) const {
        return boundary_dofs_iterator(n_dofs_, static_cast<const Derived*>(this), marker);
    }
    boundary_dofs_iterator boundary_dofs_begin() const { return boundary_dofs_begin(Boundary::All); }
    boundary_dofs_iterator boundary_dofs_end() const { return boundary_dofs_end(Boundary::All); }
    // setters
    void set_dofs_markers(const DVector<MarkerType>& dofs_markers) {
      fdapde_assert(dofs_markers.rows() == n_dofs_);
      dofs_markers_ = dofs_markers;
    }
    template <typename Iterator> void set_dofs_markers(const Iterator& begin, const Iterator& end, MarkerType marker) {
        for (auto it = begin; it != end; ++it) it->marker() = marker;
    }
    void set_dofs_markers(MarkerType marker) { set_dofs_markers(boundary_dofs_begin(), boundary_dofs_end(), marker); }
    DMatrix<double> dofs_coords() const {   // computes degrees of freedom's physical coordinates
        // allocate space
        DMatrix<double> coords;
        coords.resize(n_dofs_, TriangulationType::embed_dim);
        std::unordered_set<int> visited;
        // cycle over all mesh elements
        for (typename TriangulationType::cell_iterator e = triangulation_->cells_begin();
             e != triangulation_->cells_end(); ++e) {
            int cell_id = e->id();
            for (int j = 0; j < static_cast<const Derived&>(*this).n_dofs_per_cell(); ++j) {
                int dof = dofs_(cell_id, j);
                if (visited.find(dof) == visited.end()) {
                    // map point from reference to physical element
                    coords.row(dof) = e->J() * reference_dofs_barycentric_coords_.col(j) + e->node(0);
                    visited.insert(dof);
                }
            }
        }
        return coords;
    }
   protected:
    const TriangulationType* triangulation_;
    DMatrix<int, Eigen::RowMajor> dofs_;
    BinaryVector<fdapde::Dynamic> boundary_dofs_;   // whether the i-th dof is on boundary or not
    DVector<MarkerType> dofs_markers_;
    int n_dofs_ = 0;
    DMatrix<double> reference_dofs_barycentric_coords_;

    // returns eventual dofs inserted on the boundary
    template <bool dof_sharing, typename Iterator, typename Map>
    void local_enumerate(
      const Iterator& begin, const Iterator& end, [[maybe_unused]] Map& map, std::unordered_set<int>& boundary_dofs,
      int cell_id, int table_offset, int n_dofs_to_insert) {
        if constexpr (dof_sharing) {
            for (auto jt = begin; jt != end; ++jt) {
                if (map.find(jt->id()) == map.end()) {
                    for (int j = 0; j < n_dofs_to_insert; ++j) {
                        dofs_(cell_id, table_offset + j) = n_dofs_;
                        map[jt->id()].push_back(n_dofs_);
			if(jt->on_boundary()) boundary_dofs.insert(n_dofs_);
                        n_dofs_++;
                    }
                } else {
                    for (int j = 0; j < n_dofs_to_insert; ++j) { dofs_(cell_id, table_offset + j) = map[jt->id()][j]; }
                }
		table_offset += n_dofs_to_insert;
            }
        } else {
            for (auto jt = begin; jt != end; ++jt) {
                for (int j = 0; j < n_dofs_to_insert; ++j) {
                    if (jt->on_boundary()) boundary_dofs.insert(n_dofs_);
                    dofs_(cell_id, table_offset + j) = n_dofs_++;
		}
                table_offset += n_dofs_to_insert;
            }
        }
        return;
    }
    template <typename FEType> void enumerate([[maybe_unused]] FEType) {
        using dof_descriptor = typename FEType::cell_dof_descriptor<TriangulationType::local_dim>;
        static_assert(dof_descriptor::local_dim == TriangulationType::local_dim);
	
        dofs_.resize(triangulation_->n_cells(), dof_descriptor::n_dofs_per_cell);
	typename FEType::cell_dof_descriptor<TriangulationType::local_dim> fe;
	reference_dofs_barycentric_coords_.resize(fe.dofs_bary_coords().rows(), fe.dofs_bary_coords().cols());
	fe.dofs_bary_coords().copy_to(reference_dofs_barycentric_coords_);
        if constexpr (dof_descriptor::dof_sharing) {
            // for dof_sharing finite elements, use the geometric nodes enumeration as dof enumeration
            dofs_.leftCols(n_nodes_per_cell) = triangulation_->cells();
            n_dofs_ = triangulation_->n_nodes();
        } else {
            for (int i = 0; i < triangulation_->n_cells(); ++i) {
                for (int j = 0; j < n_nodes_per_cell; ++j) { dofs_(i, j) = n_dofs_++; }
            }
        }
        return;
    }
};

template <int EmbedDim> class DofHandler<2, EmbedDim> : public DofHandlerBase<2, EmbedDim, DofHandler<2, EmbedDim>> {
   public:
    using Base = DofHandlerBase<2, EmbedDim, DofHandler<2, EmbedDim>>;
    using TriangulationType = typename Base::TriangulationType;
    using MarkerType = typename Base::MarkerType;
    using CellType = typename Base::CellType;
    using Base::dofs_;
    using Base::n_dofs_;
    using Base::triangulation_;
  
    DofHandler() = default;
    DofHandler(const TriangulationType& triangulation) : Base(triangulation) { }

    template <typename FEType> void enumerate(FEType fe) {
        using dof_descriptor = typename FEType::cell_dof_descriptor<TriangulationType::local_dim>;
        Base::enumerate(fe);   // enumerate dofs at nodes
        n_dofs_internal_per_cell_ = dof_descriptor::n_dofs_internal;
        n_dofs_per_edge_ = dof_descriptor::n_dofs_per_edge;
        n_dofs_per_cell_ = TriangulationType::n_nodes_per_cell +
                           n_dofs_per_edge_ * TriangulationType::n_edges_per_cell + n_dofs_internal_per_cell_;
        // insert additional dofs if requested by the finite element
        std::unordered_set<int> boundary_dofs;
        if constexpr (dof_descriptor::n_dofs_per_edge > 0 || dof_descriptor::n_dofs_internal > 0) {
            constexpr int n_edges_per_cell = TriangulationType::n_edges_per_cell;
            for (typename TriangulationType::cell_iterator it = triangulation_->cells_begin();
                 it != triangulation_->cells_end(); ++it) {
                if constexpr (dof_descriptor::n_dofs_per_edge > 0) {
                    Base::template local_enumerate<dof_descriptor::dof_sharing>(
                      it->edges_begin(), it->edges_end(), edge_to_dofs_, boundary_dofs, it->id(),
                      Base::n_nodes_per_cell, n_dofs_per_edge_);
                }
                if constexpr (dof_descriptor::n_dofs_internal > 0) {
                    int table_offset = Base::n_nodes_per_cell + n_edges_per_cell * dof_descriptor::n_dofs_per_edge;
                    for (int j = 0; j < dof_descriptor::n_dofs_internal; ++j) {
                        dofs_(it->id(), table_offset + j) = n_dofs_++;
                    }
                }
            }
        }
        // update boundary
        Base::boundary_dofs_.resize(n_dofs_);
        Base::boundary_dofs_.topRows(triangulation_->n_nodes()) = triangulation_->boundary_nodes();
        for (auto it = boundary_dofs.begin(); it != boundary_dofs.end(); ++it) { Base::boundary_dofs_.set(*it); }
        return;
    }
    // getters
    int n_dofs_per_cell() const { return n_dofs_per_cell_; }
    int n_dofs_per_edge() const { return n_dofs_per_edge_; }
    int n_dofs_internal_per_cell() const { return n_dofs_internal_per_cell_; }
    const std::unordered_map<int, std::vector<int>>& edge_to_dofs() const { return edge_to_dofs_; }

    // iterator over (dof informed) edges
    class edge_iterator : public index_based_iterator<edge_iterator, typename CellType::EdgeType> {
       protected:
        using Base = index_based_iterator<edge_iterator, typename CellType::EdgeType>;
        using Base::index_;
        const DofHandler* dof_handler_;
        BinaryVector<fdapde::Dynamic> filter_;
        int n_edges_ = 0;
       public:
        edge_iterator(int index, const DofHandler* dof_handler, const BinaryVector<fdapde::Dynamic>& filter) :
            Base(index, 0, dof_handler->triangulation()->n_edges()),
            dof_handler_(dof_handler),
            filter_(filter),
            n_edges_(dof_handler->triangulation()->n_edges()) {
            for (; index_ < n_edges_ && !filter_[index_]; ++index_);
            if (index_ != n_edges_) { Base::val_ = typename CellType::EdgeType(index_, dof_handler_); }
        }
        edge_iterator(int index, const DofHandler* dof_handler) :
            edge_iterator(
              index, dof_handler, BinaryVector<fdapde::Dynamic>::Ones(dof_handler->triangulation()->n_edges())) { }
        edge_iterator& operator++() {
            // fetch next edge
            index_++;
            for (; index_ < n_edges_ && !filter_[index_]; ++index_);
            if (index_ == n_edges_) return *this;
            Base::val_ = typename CellType::EdgeType(index_, dof_handler_);
            return *this;
        }
        edge_iterator& operator--() {
            // fetch previous edge
            index_--;
            for (; index_ >= 0 && !filter_[index_]; --index_);
            if (index_ == -1) return *this;
            Base::val_ = typename CellType::EdgeType(index_, dof_handler_);
            return *this;
        }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(triangulation_->n_edges(), this); }
    // iterator over boundary edges
    struct boundary_edge_iterator : public edge_iterator {
        boundary_edge_iterator(int index, const DofHandler* dof_handler) :
            edge_iterator(index, dof_handler, dof_handler->triangulation()->boundary_edges()) { }
    };
    boundary_edge_iterator boundary_edges_begin() const { return boundary_edge_iterator(0, this); }
    boundary_edge_iterator boundary_edges_end() const {
        return boundary_edge_iterator(triangulation_->n_edges(), this);
    }
    using boundary_face_iterator = boundary_edge_iterator;
   private:
    int n_dofs_per_edge_ = 0, n_dofs_per_cell_ = 0, n_dofs_internal_per_cell_ = 0;
    std::unordered_map<int, std::vector<int>> edge_to_dofs_;   // for each edge, the dofs which are not on its nodes
};

template <> class DofHandler<3, 3> : public DofHandlerBase<3, 3, DofHandler<3, 3>> {
   private:
    using Base = DofHandlerBase<3, 3, DofHandler<3, 3>>;
    using MarkerType = typename Base::MarkerType;
    using CellType = typename Base::CellType;
    using Base::n_dofs_;
    using Base::triangulation_;
    int n_dofs_per_face_ = 0, n_dofs_per_edge_ = 0, n_dofs_per_cell_ = 0, n_dofs_internal_per_cell_ = 0;
    std::unordered_map<int, std::vector<int>> face_to_dofs_;   // for each face, its internal dofs (not nodes nor edges)
    std::unordered_map<int, std::vector<int>> edge_to_dofs_;   // for each edge, the dofs which are not on its nodes

    // basic iterator type
    template <typename Iterator, typename ValueType> class iterator : public index_based_iterator<Iterator, ValueType> {
       protected:
        using Base = index_based_iterator<Iterator, ValueType>;
        using Base::index_;
        const DofHandler* dof_handler_;
        BinaryVector<fdapde::Dynamic> filter_;
        void next_() {
            for (; index_ < Base::end_ && !filter_[index_]; ++index_);
            if (index_ == Base::end_) return;
            Base::val_ = ValueType(index_, dof_handler_);
            index_++;
        }
       public:
        iterator(
          int index, int begin, int end, const DofHandler* dof_handler, const BinaryVector<fdapde::Dynamic>& filter) :
            Base(index, begin, end), dof_handler_(dof_handler), filter_(filter) {
            next_();
        }
        iterator(int index, int begin, int end, const DofHandler* dof_handler) :
            iterator(index, begin, end, dof_handler, BinaryVector<fdapde::Dynamic>::Ones(end - begin)) { }
        Iterator& operator++() {
            next_();
            return static_cast<Iterator&>(*this);
        }
        Iterator& operator--() {
            for (; index_ >= Base::begin_ && !filter_[index_]; --index_);
            if (index_ == -1) return static_cast<Iterator&>(*this);
            Base::val_ = ValueType(index_, dof_handler_);
            index_--;
            return static_cast<Iterator&>(*this);
        }
    };
   public:
    using TriangulationType = typename Base::TriangulationType;

    DofHandler() = default;
    DofHandler(const TriangulationType& triangulation) : Base(triangulation) { }

    template <typename FEType> void enumerate(FEType fe) {
        using dof_descriptor = typename FEType::cell_dof_descriptor<TriangulationType::local_dim>;
        Base::enumerate(fe);   // enumerate dofs at nodes
        n_dofs_internal_per_cell_ = dof_descriptor::n_dofs_internal;
        n_dofs_per_face_ = dof_descriptor::n_dofs_per_face;
        n_dofs_per_edge_ = dof_descriptor::n_dofs_per_edge;
        n_dofs_per_cell_ = TriangulationType::n_nodes_per_cell +
                           n_dofs_per_face_ * TriangulationType::n_faces_per_cell +
                           n_dofs_per_edge_ * TriangulationType::n_edges_per_cell + n_dofs_internal_per_cell_;
        // insert additional dofs if requested by the finite element
        std::unordered_set<int> boundary_dofs;
        if constexpr (
          dof_descriptor::n_dofs_per_edge > 0 || dof_descriptor::n_dofs_per_face > 0 ||
          dof_descriptor::n_dofs_internal > 0) {
            constexpr int n_edges_per_cell = TriangulationType::n_edges_per_cell;
            constexpr int n_faces_per_cell = TriangulationType::n_faces_per_cell;
            for (typename TriangulationType::cell_iterator it = triangulation_->cells_begin();
                 it != triangulation_->cells_end(); ++it) {
                if constexpr (dof_descriptor::n_dofs_per_edge > 0) {
                    Base::template local_enumerate<dof_descriptor::dof_sharing>(
                      it->edges_begin(), it->edges_end(), edge_to_dofs_, boundary_dofs, it->id(),
                      Base::n_nodes_per_cell, n_dofs_per_edge_);
                }
                if constexpr (dof_descriptor::n_dofs_per_face > 0) {
                    Base::template local_enumerate<dof_descriptor::dof_sharing>(
                      it->faces_begin(), it->faces_end(), face_to_dofs_, boundary_dofs, it->id(),
                      Base::n_nodes_per_cell + dof_descriptor::n_dofs_per_edge * n_edges_per_cell, n_dofs_per_face_);
                }
                if constexpr (dof_descriptor::n_dofs_internal > 0) {
                    int table_offset = Base::n_nodes_per_cell + n_edges_per_cell * dof_descriptor::n_dofs_per_edge +
                                       n_faces_per_cell * dof_descriptor::n_dofs_per_face;
                    for (int j = 0; j < dof_descriptor::n_dofs_internal; ++j) {
                        dofs_(it->id(), table_offset + j) = n_dofs_++;
                    }
                }
            }
        }
        // update boundary
        Base::boundary_dofs_.resize(n_dofs_);
        Base::boundary_dofs_.topRows(triangulation_->n_nodes()) = triangulation_->boundary_nodes();
        for (auto it = boundary_dofs.begin(); it != boundary_dofs.end(); ++it) { Base::boundary_dofs_.set(*it); }
        return;
    }
    // getters
    int n_dofs_per_cell() const { return n_dofs_per_cell_; }
    int n_dofs_per_edge() const { return n_dofs_per_edge_; }
    int n_dofs_per_face() const { return n_dofs_per_face_; }
    int n_dofs_internal_per_cell() const { return n_dofs_internal_per_cell_; }
    const std::unordered_map<int, std::vector<int>>& edge_to_dofs() const { return edge_to_dofs_; }
    const std::unordered_map<int, std::vector<int>>& face_to_dofs() const { return face_to_dofs_; }

    // iterators over edges
    struct edge_iterator : public iterator<edge_iterator, typename CellType::EdgeType> {
        edge_iterator(int index, const DofHandler* dof_handler, const BinaryVector<fdapde::Dynamic>& filter) :
            iterator<edge_iterator, typename CellType::EdgeType>(
              index, 0, dof_handler->triangulation()->n_edges(), dof_handler, filter) { }
        edge_iterator(int index, const DofHandler* dof_handler) :
            iterator<edge_iterator, typename CellType::EdgeType>(
              index, 0, dof_handler->triangulation()->n_edges(), dof_handler) { }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(triangulation_->n_edges(), this); }
    // iterators over faces
    struct face_iterator : public iterator<face_iterator, typename CellType::FaceType> {
        face_iterator(int index, const DofHandler* dof_handler, const BinaryVector<fdapde::Dynamic>& filter) :
            iterator<face_iterator, typename CellType::FaceType>(
              index, 0, dof_handler->triangulation()->n_faces(), dof_handler, filter) { }
        face_iterator(int index, const DofHandler* dof_handler) :
            iterator<face_iterator, typename CellType::FaceType>(
              index, 0, dof_handler->triangulation()->n_faces(), dof_handler) { }
    };
    // iterator over boundary faces
    struct boundary_face_iterator : public face_iterator {
        boundary_face_iterator(int index, const DofHandler* dof_handler) :
            face_iterator(index, dof_handler, dof_handler->triangulation()->boundary_faces()) { }
    };
    boundary_face_iterator boundary_faces_begin() const { return boundary_face_iterator(0, this); }
    boundary_face_iterator boundary_faces_end() const {
        return boundary_face_iterator(triangulation_->n_faces(), this);
    }
};

// template <int EmbedDim> class DofHandler<1, EmbedDim> : public DofHandlerBase<1, EmbedDim, DofHandler<1, EmbedDim>> {
//    private:
//     using Base = DofHandlerBase<1, EmbedDim, DofHandler<1, EmbedDim>>;
//     using TriangulationType = typename Base::TriangulationType;
//     using MarkerType = typename Base::MarkerType;
//     using CellType = DofHandlerCell<Segment<Triangulation<1, EmbedDim>>>;
//     using Base::n_dofs_;
//     using Base::triangulation_;
//     int n_dofs_per_cell_ = 0, n_dofs_internal_per_cell_ = 0;
//    public:
//     DofHandler() = default;
//     DofHandler(const TriangulationType& triangulation) : Base(triangulation) { }

//     template <typename FEType> void enumerate(FEType fe) {
//         Base::enumerate(fe);   // enumerate dofs at nodes
//         n_dofs_internal_per_cell_ = FEType::n_dofs_internal;
//         n_dofs_per_cell_ = TriangulationType::n_nodes_per_cell + n_dofs_internal_per_cell_;
//         // insert additional dofs if requested by the finite element
//         if constexpr (FEType::n_dofs_internal > 0) {
//             for (typename TriangulationType::cell_iterator it = triangulation_->cells_begin();
//                  it != triangulation_->cells_end(); ++it) {
//                 for (int j = 0; j < FEType::n_dofs_internal; ++j) { dofs_(it->id(), n_nodes_per_cell + j) = n_dofs_++; }
//             }
//         }
//         // update boundary
//         Base::boundary_dofs_.resize(n_dofs_);
//         Base::boundary_dofs_.topRows(triangulation_->n_nodes()) = triangulation_->boundary_nodes();
//         return;
//     }
//     // getters
//     int n_dofs_per_cell() const { return n_dofs_per_cell_; }
//     int n_dofs_internal_per_cell() const { return n_dofs_internal_per_cell_; }
// };
  
// template argument deduction guide
template <int LocalDim, int EmbedDim> DofHandler(Triangulation<LocalDim, EmbedDim>) -> DofHandler<LocalDim, EmbedDim>;

}   // namespace fdapde

#endif   // __DOF_HANDLER_H__
