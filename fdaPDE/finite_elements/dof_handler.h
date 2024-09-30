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
#include "dof_constraints.h"

namespace fdapde {

template <int LocalDim, int EmbedDim> class DofHandler;
template <int LocalDim, int EmbedDim, typename Derived> class DofHandlerBase {
   public:
    using TriangulationType = Triangulation<LocalDim, EmbedDim>;
    using CellType = std::conditional_t<LocalDim == 2, DofTriangle<Derived>, DofTetrahedron<Derived>>;
    static constexpr int n_nodes_per_cell = TriangulationType::n_nodes_per_cell;
    static constexpr int local_dim = TriangulationType::local_dim;
    static constexpr int embed_dim = TriangulationType::embed_dim;
    DofHandlerBase() = default;
    DofHandlerBase(const TriangulationType& triangulation) :
        triangulation_(&triangulation), dof_constraints_(static_cast<Derived&>(*this)) { }
    // getters
    CellType cell(int id) const { return CellType(id, static_cast<const Derived*>(this)); }
    const DMatrix<int, Eigen::RowMajor>& dofs() const { return dofs_; }
    int n_dofs() const { return n_dofs_; }
    int n_unique_dofs() const { return n_unique_dofs_; }
    bool is_dof_on_boundary(int i) const { return boundary_dofs_[i]; }
    Eigen::Map<const DVector<int>> dofs_markers() const {
        return Eigen::Map<const DVector<int>>(dofs_markers_.data(), n_dofs_, 1);
    }
    int dof_marker(int dof) const { return dofs_markers_[dof]; }
    const TriangulationType* triangulation() const { return triangulation_; }
    int n_boundary_dofs() const { return boundary_dofs_.count(); }
    int n_boundary_dofs(int marker) const {
        int i = 0, sum = 0;
        for (int dof_marker : dofs_markers_) { sum += (dof_marker == marker && boundary_dofs_[i++]) ? 1 : 0; }
        return sum;
    }
    std::vector<int> filter_dofs_by_marker(int marker) const {
        std::vector<int> result;
        for (int i = 0; i < n_dofs_; ++i) {
            if (dofs_markers_[i] == marker) result.push_back(i);
        }
        return result;
    }
    DVector<int> active_dofs(int cell_id) const { return dofs_.row(cell_id); }   // dofs located on cell with ID cell_id
    operator bool() const { return n_dofs_ != 0; }
    Eigen::Map<const DVector<int>> dofs_to_cell() const {
        return Eigen::Map<const DVector<int>>(dofs_to_cell_.data(), n_dofs_, 1);
    }

    // iterates over geometric cells coupled with dofs informations (possibly filtered by marker)
    class cell_iterator : public internals::filtering_iterator<cell_iterator, CellType> {
        using Base = internals::filtering_iterator<cell_iterator, CellType>;
        using Base::index_;
        friend Base;
        const Derived* dof_handler_;
        int marker_;
        cell_iterator& operator()(int i) {
            Base::val_ = dof_handler_->cell(i);
            return *this;
        }
       public:
        cell_iterator() = default;
        cell_iterator(int index, const Derived* dof_handler, const BinaryVector<fdapde::Dynamic>& filter, int marker) :
            Base(index, 0, dof_handler->triangulation()->n_cells(), filter),
            dof_handler_(dof_handler),
            marker_(marker) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        cell_iterator(int index, const Derived* dof_handler, int marker) :
            cell_iterator(
              index, dof_handler,
              marker == TriangulationAll ?
                BinaryVector<fdapde::Dynamic>::Ones(dof_handler->triangulation()->n_cells()) :   // apply no filter
                fdapde::make_binary_vector(
                  dof_handler->triangulation()->cells_markers().begin(),
                  dof_handler->triangulation()->cells_markers().end(), marker),
              marker) { }
        int marker() const { return marker_; }
    };  
    cell_iterator cells_begin(int marker = TriangulationAll) const {
        const std::vector<int>& cells_markers = triangulation_->cells_markers();
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers.size() != 0));
        return cell_iterator(0, static_cast<const Derived*>(this), marker);
    }
    cell_iterator cells_end(int marker = TriangulationAll) const {
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && triangulation_->cells_markers().size() != 0));
        return cell_iterator(triangulation_->n_cells(), static_cast<const Derived*>(this), marker);
    }
    class BoundaryDofType {
        int id_;
        const Derived* dof_handler_;
       public:
        BoundaryDofType() = default;
        BoundaryDofType(int id, const Derived* dof_handler) : id_(id), dof_handler_(dof_handler) { }
        int id() const { return id_; }
        int marker() const { return dof_handler_->dofs_markers_[id_]; }
        SVector<embed_dim> coord() const {
	    int cell_id = dof_handler_->dofs_to_cell()[id_];   // id of cell containing this dof
            int j = 0;   // local dof numbering
            for (; j < dof_handler_->n_dofs_per_cell() && dof_handler_->dofs()(cell_id, j) != id_; ++j);
            typename Derived::CellType cell = dof_handler_->cell(cell_id);
	    // compute dof physical coordinate
            return cell.J() * dof_handler_->reference_dofs_barycentric_coords_.rightCols(local_dim).row(j).transpose() +
                   cell.node(0);
        }
    };
    class boundary_dofs_iterator : public internals::filtering_iterator<boundary_dofs_iterator, BoundaryDofType> {
        using Base = internals::filtering_iterator<boundary_dofs_iterator, BoundaryDofType>;
        using Base::index_;
        friend Base;
        const Derived* dof_handler_;
        int marker_;
        boundary_dofs_iterator& operator()(int i) {
            Base::val_ = BoundaryDofType(i, dof_handler_);
            return *this;
        }
       public:
        boundary_dofs_iterator(
          int index, const Derived* dof_handler, const BinaryVector<fdapde::Dynamic>& filter, int marker) :
            Base(index, 0, dof_handler->n_dofs(), filter), dof_handler_(dof_handler), marker_(marker) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        boundary_dofs_iterator(int index, const Derived* dof_handler, int marker) :   // filter boundary dofs by marker
            boundary_dofs_iterator(
              index, dof_handler,
              marker == BoundaryAll ? dof_handler->boundary_dofs_ :
                                      dof_handler->boundary_dofs_ &
                                        fdapde::make_binary_vector(
                                          dof_handler->dofs_markers_.begin(), dof_handler->dofs_markers_.end(), marker),
              marker) { }
        int marker() const { return marker_; }
    };
    boundary_dofs_iterator boundary_dofs_begin(int marker = BoundaryAll) const {
        return boundary_dofs_iterator(0, static_cast<const Derived*>(this), marker);
    }
    boundary_dofs_iterator boundary_dofs_end(int marker = BoundaryAll) const {
      return boundary_dofs_iterator(n_dofs_, static_cast<const Derived*>(this), marker);
    }
    // dofs constaints handling
    template <typename... Data> void set_dirichlet_constraint(int on, Data&&... g) {
        fdapde_assert(sizeof...(Data) == derived().dof_multiplicity());
        dof_constraints_.set_dirichlet_constraint(on, g...);
    }
    template <typename... Data> void set_dirichlet_constaint(Data&&... g) {
        set_dirichlet_constraint(BoundaryAll, g...);
    }
    template <typename SystemMatrix, typename SystemRhs>
    void enforce_constraints(SystemMatrix&& A, SystemRhs&& b) const {
        dof_constraints_.enforce_constraints(A, b);
    }
    void enforce_constraints(SpMatrix<double>& A) const { dof_constraints_.enforce_constraints(A); }
    void enforce_constraints(DVector<double>& b)  const { dof_constraints_.enforce_constraints(b); }
    DMatrix<double> dofs_coords() const {   // computes degrees of freedom's physical coordinates
        // allocate space (just for unique dofs, i.e., without considering any dof multiplicity)
        DMatrix<double> coords;
        coords.resize(n_unique_dofs_, TriangulationType::embed_dim);
        std::unordered_set<int> visited;
        // cycle over all mesh elements
        for (typename TriangulationType::cell_iterator cell = triangulation_->cells_begin();
             cell != triangulation_->cells_end(); ++cell) {
            int cell_id = cell->id();
            for (int j = 0; j < derived().n_dofs_per_cell(); ++j) {
                int dof = dofs_(cell_id, j);
                if (visited.find(dof) == visited.end()) {
                    // map point from reference to physical element
                    coords.row(dof) =
                      cell->J() * reference_dofs_barycentric_coords_.rightCols(local_dim).row(j).transpose() +
                      cell->node(0);
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
    std::vector<int> dofs_to_cell_;                 // for each dof, the id of (one of) the cell containing it
    DofConstraints<local_dim, embed_dim> dof_constraints_;
    std::vector<int> dofs_markers_;
    int n_dofs_ = 0, n_unique_dofs_ = 0;
    DMatrix<double> reference_dofs_barycentric_coords_;

    // local enumeration of dofs on cell
    template <bool dof_sharing, typename Iterator, typename Map>
    void local_enumerate(
      const Iterator& begin, const Iterator& end, [[maybe_unused]] Map& map,
      std::unordered_set<std::pair<int, int>, fdapde::pair_hash>& boundary_dofs, int cell_id, int table_offset,
      int n_dofs_to_insert) {
        if constexpr (dof_sharing) {
            for (auto jt = begin; jt != end; ++jt) {
                if (map.find(jt->id()) == map.end()) {
                    for (int j = 0; j < n_dofs_to_insert; ++j) {
                        dofs_(cell_id, table_offset + j) = n_dofs_;
                        map[jt->id()].push_back(n_dofs_);
                        if (jt->on_boundary()) {
                            // inserted boundary dofs are marked with the corresponding geometrical boundary marker
                            boundary_dofs.insert({n_dofs_, jt->marker()});
                        }
                        n_dofs_++;
                        dofs_to_cell_.push_back(cell_id);
                    }
                } else {
                    for (int j = 0; j < n_dofs_to_insert; ++j) { dofs_(cell_id, table_offset + j) = map[jt->id()][j]; }
                }
		table_offset += n_dofs_to_insert;
            }
        } else {
            for (auto jt = begin; jt != end; ++jt) {
                for (int j = 0; j < n_dofs_to_insert; ++j) {
                    if (jt->on_boundary()) { boundary_dofs.insert({n_dofs_, jt->marker()}); }
                    dofs_(cell_id, table_offset + j) = n_dofs_++;
                    dofs_to_cell_.push_back(cell_id);
                }
                table_offset += n_dofs_to_insert;
            }
        }
        return;
    }
    template <typename FEType> void enumerate([[maybe_unused]] FEType) {
        using dof_descriptor = typename FEType::cell_dof_descriptor<TriangulationType::local_dim>;
        fdapde_static_assert(
          dof_descriptor::local_dim == TriangulationType::local_dim, YOU_PROVIDED_A_WRONG_FINITE_ELEMENT_DESCRIPTOR);
        fdapde_static_assert(
          dof_descriptor::n_dofs_per_cell > 0 && dof_descriptor::dof_multiplicity > 0,
          FINITE_ELEMENT_DESCRIPTION_REQUESTS_THE_INSERTION_OF_AT_LEAST_ONE_DEGREE_OF_FREEDOM_PER_CELL);
        dofs_.resize(triangulation_->n_cells(), dof_descriptor::n_dofs_per_cell * dof_descriptor::dof_multiplicity);
	// copy coordinates of dofs defined on reference unit simplex
	typename FEType::cell_dof_descriptor<TriangulationType::local_dim> fe;
	reference_dofs_barycentric_coords_.resize(fe.dofs_bary_coords().rows(), fe.dofs_bary_coords().cols());
	fe.dofs_bary_coords().copy_to(reference_dofs_barycentric_coords_);
	// start enumeration at geometrical nodes
	static constexpr int n_dofs_at_nodes = dof_descriptor::n_dofs_per_node * TriangulationType::n_nodes_per_cell;
	const int n_cells = triangulation_->n_cells();
	n_dofs_ = 0;
        if constexpr (dof_descriptor::dof_sharing) {
            if constexpr (dof_descriptor::n_dofs_per_node > 0) {
                fdapde_static_assert(
                  dof_descriptor::n_dofs_per_node == 1,
                  DOF_SHARING_ON_CELLS_NODES_REQUIRES_ONE_DEGREES_OF_FREEDOM_PER_NODE);
                // for dof_sharing finite elements, use the geometric nodes enumeration as dof enumeration
                dofs_.leftCols(n_dofs_at_nodes) = triangulation_->cells();
                n_dofs_ = triangulation_->n_nodes();
            }
        } else {
            for (int i = 0; i < n_cells; ++i) {
                for (int j = 0; j < n_dofs_at_nodes; ++j) { dofs_(i, j) = n_dofs_++; }
            }
        }
	// update dof to cell mapping
        dofs_to_cell_.resize(n_dofs_);
        for (int i = 0; i < n_cells; ++i) {
            for (int j = 0; j < n_dofs_at_nodes; ++j) { dofs_to_cell_[dofs_(i, j)] = i; }
        }
        dofs_markers_ = std::vector<int>(n_dofs_, Unmarked);
        return;
    }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};
  
template <int EmbedDim> class DofHandler<2, EmbedDim> : public DofHandlerBase<2, EmbedDim, DofHandler<2, EmbedDim>> {
   public:
    using Base = DofHandlerBase<2, EmbedDim, DofHandler<2, EmbedDim>>;
    using TriangulationType = typename Base::TriangulationType;
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
	n_dofs_per_node_ = dof_descriptor::n_dofs_per_node;
        n_dofs_per_edge_ = dof_descriptor::n_dofs_per_edge;
        n_dofs_per_cell_ = n_dofs_per_node_ * TriangulationType::n_nodes_per_cell +
                           n_dofs_per_edge_ * TriangulationType::n_edges_per_cell + n_dofs_internal_per_cell_;
	dof_multiplicity_ = dof_descriptor::dof_multiplicity;
        // move geometrical markers on boundary edges to dof markers on nodes. high labeled nodes have higher priority
        if constexpr (dof_descriptor::n_dofs_per_node > 0) {
            for (typename TriangulationType::edge_iterator it = triangulation_->boundary_edges_begin();
                 it != triangulation_->boundary_edges_end(); ++it) {
                int marker = it->marker();
                for (int node_id : it->node_ids()) {
                    // give priority to highly marked edges
                    if (marker > Base::dofs_markers_[node_id]) { Base::dofs_markers_[node_id] = marker; }
                }
            }
        }
        // insert additional dofs if requested by the finite element
	static constexpr int n_dofs_at_nodes = dof_descriptor::n_dofs_per_node * TriangulationType::n_nodes_per_cell;
        std::unordered_set<std::pair<int, int>, fdapde::pair_hash> boundary_dofs;
        if constexpr (dof_descriptor::n_dofs_per_edge > 0 || dof_descriptor::n_dofs_internal > 0) {
            constexpr int n_edges_per_cell = TriangulationType::n_edges_per_cell;
            auto edge_pattern =
              cexpr::combinations<TriangulationType::n_nodes_per_edge, TriangulationType::n_nodes_per_cell>();

            for (typename TriangulationType::cell_iterator it = triangulation_->cells_begin();
                 it != triangulation_->cells_end(); ++it) {
                int cell_id = it->id();
                if constexpr (dof_descriptor::n_dofs_per_edge > 0) {
                    Base::template local_enumerate<dof_descriptor::dof_sharing>(
                      it->edges_begin(), it->edges_end(), edge_to_dofs_, boundary_dofs, cell_id, n_dofs_at_nodes,
                      n_dofs_per_edge_);

                    if constexpr (dof_descriptor::n_dofs_per_edge > 1) {
                        // reorder dofs to preserve numbering monotonicity on edge
                        int offset = 0;
                        for (int i = 0; i < edge_pattern.rows(); ++i) {   // cycle on edges
                            if (dofs_(cell_id, edge_pattern(i, 0)) > dofs_(cell_id, edge_pattern(i, 1))) {
                                // reverse in-place
                                offset = n_dofs_at_nodes + i * dof_descriptor::n_dofs_per_edge;
                                int k = 0, h = dof_descriptor::n_dofs_per_edge - 1;
                                int tmp;
                                while (k < h) {
                                    tmp = dofs_(cell_id, offset + h);
                                    dofs_(cell_id, offset + h) = dofs_(cell_id, offset + k);
                                    dofs_(cell_id, offset + k) = tmp;
                                    h--;
                                    k++;
                                }
                            }
                        }
                    }
                }
                if constexpr (dof_descriptor::n_dofs_internal > 0) {
                    // internal dofs are never shared
                    int table_offset = n_dofs_at_nodes + n_edges_per_cell * dof_descriptor::n_dofs_per_edge;
                    for (int j = 0; j < dof_descriptor::n_dofs_internal; ++j) {
                        dofs_(cell_id, table_offset + j) = n_dofs_++;
                        Base::dofs_to_cell_.push_back(cell_id);
                    }
		    // if the internal dofs are the unique inserted dofs, we must move the boundary marker to such dofs
		    // this logic should be triggered only from order 0 elements
                    if constexpr (dof_descriptor::n_dofs_internal == dof_descriptor::n_dofs_per_cell) {
                        if (it->on_boundary()) {
                            // search for boundary marker
                            int marker = Unmarked;
                            for (auto jt = it->edges_begin(); jt != it->edges_end(); ++jt) {
                                if (jt->on_boundary() && jt->marker() > marker) { marker = jt->marker(); }
                            }
                            for (int j = 0; j < dof_descriptor::n_dofs_internal; ++j) {
                                // all internal nodes share the same marker
                                boundary_dofs.insert({dofs_(cell_id, table_offset + j), marker});
                            }
                        }
                    }
                }
            }
        }
        Base::n_unique_dofs_ = n_dofs_;
        // update boundary
        Base::boundary_dofs_.resize(n_dofs_ * dof_descriptor::dof_multiplicity);
        if constexpr (dof_descriptor::n_dofs_per_node > 0) {   // inherit boundary description from geometry
            Base::boundary_dofs_.topRows(triangulation_->n_nodes()) = triangulation_->boundary_nodes();
        }
        if constexpr (dof_descriptor::n_dofs_per_edge > 0 || dof_descriptor::n_dofs_internal > 0) {
            Base::dofs_markers_.resize(n_dofs_, Unmarked);
            for (auto it = boundary_dofs.begin(); it != boundary_dofs.end(); ++it) {
                Base::boundary_dofs_.set(it->first);
                Base::dofs_markers_[it->first] = it->second;
            }
        }
        // if dof_multiplicity is higher than one, replicate the computed dof numbering adding n_dofs_ to each dof
        if constexpr (dof_descriptor::dof_multiplicity > 1) {
            Base::dofs_to_cell_.resize(n_dofs_ * dof_descriptor::dof_multiplicity);
            Base::dofs_markers_.resize(n_dofs_ * dof_descriptor::dof_multiplicity);
            for (int i = 1; i < dof_descriptor::dof_multiplicity; ++i) {
                dofs_.middleCols(i * dof_descriptor::n_dofs_per_cell, dof_descriptor::n_dofs_per_cell) =
                  dofs_.leftCols(dof_descriptor::n_dofs_per_cell).array() + (i * n_dofs_);
                Base::boundary_dofs_.middleRows(i * n_dofs_, n_dofs_) = Base::boundary_dofs_.topRows(n_dofs_);
                for (int j = 0; j < n_dofs_; ++j) {
                    Base::dofs_to_cell_[j + i * n_dofs_] = Base::dofs_to_cell_[j];
                    Base::dofs_markers_[j + i * n_dofs_] = Base::dofs_markers_[j];
                }
                if constexpr (dof_descriptor::n_dofs_per_edge > 0) {
                    for (auto& [edge_id, dof_ids] : edge_to_dofs_) {
                        for (int dof : dof_ids) { dof_ids.push_back(dof + i * n_dofs_); }
                    }
                }
            }
	    n_dofs_ = n_dofs_ * dof_descriptor::dof_multiplicity;
        }
        return;
    }
    // getters
    int n_dofs_per_cell() const { return n_dofs_per_cell_; }
    int n_dofs_per_edge() const { return n_dofs_per_edge_; }
    int n_dofs_per_node() const { return n_dofs_per_node_; }
    int n_dofs_internal_per_cell() const { return n_dofs_internal_per_cell_; }
    int dof_multiplicity() const { return dof_multiplicity_; }
    const std::unordered_map<int, std::vector<int>>& edge_to_dofs() const { return edge_to_dofs_; }

    // iterator over (dof informed) edges
    class edge_iterator : public internals::filtering_iterator<edge_iterator, typename CellType::EdgeType> {
       protected:
        using Base = internals::filtering_iterator<edge_iterator, typename CellType::EdgeType>;
        using Base::index_;
        friend Base;
        const DofHandler* dof_handler_;
        edge_iterator& operator()(int i) {
            Base::val_ = typename CellType::EdgeType(i, dof_handler_);
            return *this;
        }
       public:
        edge_iterator(int index, const DofHandler* dof_handler, const BinaryVector<fdapde::Dynamic>& filter) :
            Base(index, 0, dof_handler->triangulation()->n_edges(), filter), dof_handler_(dof_handler) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
      edge_iterator(int index, const DofHandler* dof_handler) : // apply no filter
            edge_iterator(
              index, dof_handler, BinaryVector<fdapde::Dynamic>::Ones(dof_handler->triangulation()->n_edges())) { }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(triangulation_->n_edges(), this); }
    // iterator over boundary edges
    struct boundary_edge_iterator : public edge_iterator {
       private:
        int marker_;
       public:
        boundary_edge_iterator(int index, const DofHandler* dof_handler) :
            edge_iterator(index, dof_handler, dof_handler->triangulation()->boundary_edges()), marker_(BoundaryAll) { }
        boundary_edge_iterator(
          int index, const DofHandler* dof_handler, int marker) :   // filter boundary edges by marker
            edge_iterator(
              index, dof_handler,
              marker == BoundaryAll ? dof_handler->triangulation()->boundary_edges() :
                                      dof_handler->triangulation()->boundary_edges() &
                                        fdapde::make_binary_vector(
                                          dof_handler->triangulation()->edges_markers().begin(),
                                          dof_handler->triangulation()->edges_markers().end(), marker)) { }
        int marker() const { return marker_; }
    };
    boundary_edge_iterator boundary_edges_begin() const { return boundary_edge_iterator(0, this); }
    boundary_edge_iterator boundary_edges_end() const {
        return boundary_edge_iterator(triangulation_->n_edges(), this);
    }
    using boundary_iterator = boundary_edge_iterator;
   private:
    int n_dofs_per_node_ = 0, n_dofs_per_edge_ = 0, n_dofs_per_cell_ = 0, n_dofs_internal_per_cell_ = 0;
    int dof_multiplicity_ = 0;
    std::unordered_map<int, std::vector<int>> edge_to_dofs_;   // for each edge, the dofs which are not on its nodes
};

template <> class DofHandler<3, 3> : public DofHandlerBase<3, 3, DofHandler<3, 3>> {
   private:
    using Base = DofHandlerBase<3, 3, DofHandler<3, 3>>;
    using CellType = typename Base::CellType;
    using Base::n_dofs_;
    using Base::triangulation_;
    int n_dofs_per_face_ = 0, n_dofs_per_edge_ = 0, n_dofs_per_cell_ = 0, n_dofs_internal_per_cell_ = 0;
    std::unordered_map<int, std::vector<int>> face_to_dofs_;   // for each face, its internal dofs (not nodes nor edges)
    std::unordered_map<int, std::vector<int>> edge_to_dofs_;   // for each edge, the dofs which are not on its nodes

    // basic iterator type
    template <typename Iterator, typename ValueType> class iterator : public internals::index_iterator<Iterator, ValueType> {
       protected:
        using Base = internals::index_iterator<Iterator, ValueType>;
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
    struct boundary_iterator : public face_iterator {
        boundary_iterator(int index, const DofHandler* dof_handler) :
            face_iterator(index, dof_handler, dof_handler->triangulation()->boundary_faces()) { }
    };
    boundary_iterator boundary_faces_begin() const { return boundary_iterator(0, this); }
    boundary_iterator boundary_faces_end() const {
        return boundary_iterator(triangulation_->n_faces(), this);
    }
};

// template <int EmbedDim> class DofHandler<1, EmbedDim> : public DofHandlerBase<1, EmbedDim, DofHandler<1, EmbedDim>> {
//    private:
//     using Base = DofHandlerBase<1, EmbedDim, DofHandler<1, EmbedDim>>;
//     using TriangulationType = typename Base::TriangulationType;
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
