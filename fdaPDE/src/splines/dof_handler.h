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

#ifndef __FDAPDE_SP_DOF_HANDLER_H__
#define __FDAPDE_SP_DOF_HANDLER_H__

#include "header_check.h"

namespace fdapde {

template <int LocalDim, int EmbedDim, typename DiscretizationCategory> class DofHandler;
template <> class DofHandler<1, 1, spline_tag> {
   public:
    using TriangulationType = Triangulation<1, 1>;
    static constexpr int local_dim = TriangulationType::local_dim;
    static constexpr int embed_dim = TriangulationType::embed_dim;

    // a geometrical segment with attached dofs
    struct CellType : public Segment<TriangulationType> {
        using Base = Segment<TriangulationType>;
        const DofHandler* dof_handler_;
       public:
        static constexpr int local_dim = 1;
        static constexpr int embed_dim = 1;

        CellType() : dof_handler_(nullptr) { }
        CellType(int cell_id, const DofHandler* dof_handler) :
            Base(cell_id, dof_handler->triangulation()), dof_handler_(dof_handler) { }
        std::vector<int> dofs() const { return dof_handler_->active_dofs(Base::id()); }
        std::vector<int> dofs_markers() const {
            std::vector<int> dofs_ = dofs();
            std::vector<int> dofs_markers_(dofs_.size());
            for (int i = 0, n = dofs_.size(); i < n; ++i) { dofs_markers_[i] = dof_handler_->dof_marker(dofs_[i]); }
	    return dofs_markers_;
        }
        BinaryVector<Dynamic> boundary_dofs() const {
            std::vector<int> dofs_ = dofs();
            BinaryVector<Dynamic> boundary(dofs_.size());
            int i = 0;
            for (int dof : dofs_) {
                if (dof_handler_->is_dof_on_boundary(dof)) boundary.set(i);
                ++i;
            }
            return boundary;
        }
    };
    // constructor
    DofHandler() = default;
    DofHandler(const TriangulationType& triangulation) : triangulation_(std::addressof(triangulation)), degree_() { }
    // getters
    CellType cell(int id) const { return CellType(id, this); }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> dofs() const {   // dofs active on cell
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          dofs_.data(), dofs_.size() / 2, 2);
    }
    int n_dofs() const { return n_dofs_; }
    int n_dofs_per_cell() const { return n_dofs_per_cell_; }
    bool is_dof_on_boundary(int i) const { return boundary_dofs_[i]; }
    const std::vector<int>& dofs_markers() const { return dofs_markers_; }
    int dof_marker(int dof) const { return dofs_markers_[dof]; }
    const TriangulationType* triangulation() const { return triangulation_; }
    const std::vector<double>& dofs_coords() const { return dofs_coords_; }
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
    // In any given knot span [u_i, u_{i+1}) at most p+1 basis functions are non zero, namely N_{i-p,p}, ..., N_{i,p}
    // (property P2.2, pag 55, Piegl, L., & Tiller, W. (2012). The NURBS book. Springer Science & Business Media.)
    std::vector<int> active_dofs(int i) const {
        std::vector<int> dofs;
        for (int j = 0; j < degree_ + 1; ++j) { dofs.push_back(i + j); }
        return dofs;
    }
    template <typename ContainerT> void active_dofs(int i, ContainerT& dst) const { dst = active_dofs(i); }
    operator bool() const { return n_dofs_ != 0; }

    // iterates over geometric cells coupled with dofs informations (possibly filtered by marker)
    class cell_iterator : public internals::filtering_iterator<cell_iterator, CellType> {
        using Base = internals::filtering_iterator<cell_iterator, CellType>;
        using Base::index_;
        friend Base;
        const DofHandler* dof_handler_;
        int marker_;
        cell_iterator& operator()(int i) {
            Base::val_ = dof_handler_->cell(i);
            return *this;
        }
       public:
        cell_iterator() = default;
        cell_iterator(
          int index, const DofHandler* dof_handler, const BinaryVector<Dynamic>& filter, int marker) :
            Base(index, 0, dof_handler->triangulation()->n_cells(), filter),
            dof_handler_(dof_handler),
            marker_(marker) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        cell_iterator(int index, const DofHandler* dof_handler, int marker) :
            cell_iterator(
              index, dof_handler,
              marker == TriangulationAll ?
                BinaryVector<Dynamic>::Ones(dof_handler->triangulation()->n_cells()) :   // apply no filter
                make_binary_vector(
                  dof_handler->triangulation()->cells_markers().begin(),
                  dof_handler->triangulation()->cells_markers().end(), marker),
              marker) { }
        int marker() const { return marker_; }
    };  
    cell_iterator cells_begin(int marker = TriangulationAll) const {
        const std::vector<int>& cells_markers = triangulation_->cells_markers();
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers.size() != 0));
        return cell_iterator(0, this, marker);
    }
    cell_iterator cells_end(int marker = TriangulationAll) const {
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && triangulation_->cells_markers().size() != 0));
        return cell_iterator(triangulation_->n_cells(), this, marker);
    }

    class BoundaryDofType {
        int id_;
        const DofHandler* dof_handler_;
       public:
        BoundaryDofType() = default;
        BoundaryDofType(int id, const DofHandler* dof_handler) : id_(id), dof_handler_(dof_handler) { }
        int id() const { return id_; }
        int marker() const { return dof_handler_->dofs_markers_[id_]; }
        Eigen::Matrix<double, embed_dim, 1> coord() const {
            return Eigen::Matrix<double, embed_dim, 1>(dof_handler_->dofs_coords_[id_]);
        }
    };
    class boundary_dofs_iterator : public internals::filtering_iterator<boundary_dofs_iterator, BoundaryDofType> {
        using Base = internals::filtering_iterator<boundary_dofs_iterator, BoundaryDofType>;
        using Base::index_;
        friend Base;
        const DofHandler* dof_handler_;
        int marker_;
        boundary_dofs_iterator& operator()(int i) {
            Base::val_ = BoundaryDofType(i, dof_handler_);
            return *this;
        }
       public:
        boundary_dofs_iterator(
          int index, const DofHandler* dof_handler, const BinaryVector<Dynamic>& filter, int marker) :
            Base(index, 0, dof_handler->n_dofs(), filter), dof_handler_(dof_handler), marker_(marker) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        // filter boundary dofs by marker
        boundary_dofs_iterator(int index, const DofHandler* dof_handler, int marker) :
            boundary_dofs_iterator(
              index, dof_handler,
              marker == BoundaryAll ? dof_handler->boundary_dofs_ :
                                      dof_handler->boundary_dofs_ &
                                        make_binary_vector(
                                          dof_handler->dofs_markers_.begin(), dof_handler->dofs_markers_.end(), marker),
              marker) { }
        int marker() const { return marker_; }
    };
    boundary_dofs_iterator boundary_dofs_begin(int marker = BoundaryAll) const {
        return boundary_dofs_iterator(0, this, marker);
    }
    boundary_dofs_iterator boundary_dofs_end(int marker = BoundaryAll) const {
        return boundary_dofs_iterator(n_dofs_, this, marker);
    }

    template <typename SpType> void enumerate(SpType&& sp) {
        n_dofs_ = sp.size();
        degree_ = sp.degree();
        n_dofs_per_cell_ = degree_ + 1;
        dofs_coords_.resize(n_dofs_);
        for (int i = 0; i < n_dofs_; ++i) { dofs_coords_[i] = sp[i].knot(); }
        int n_cells = triangulation()->n_cells();
        for (int j = 0; j < n_cells; ++j) {
            dofs_.push_back(j);
	    dofs_.push_back(j + degree_);
        }	
        // Regardless of the number of physical dofs at the interval boundary, only the basis functions associated with
        // the first and last dofs are non-zero at the boundary nodes. Hence, we treat only these dofs as boundary dofs
	boundary_dofs_.resize(n_dofs_);
        boundary_dofs_.set(0);
        boundary_dofs_.set(dofs_.back());
	// inherit markers from geometry
        dofs_markers_ = triangulation_->nodes_markers();
        return;
    }
   private:
    std::vector<double> dofs_coords_;       // physical knots vector
    BinaryVector<Dynamic> boundary_dofs_;   // whether the i-th dof is on boundary or not
    std::vector<int> dofs_;
    int n_dofs_per_cell_ = 0, n_dofs_ = 0;
    std::vector<int> dofs_markers_;
    const TriangulationType* triangulation_;
    int degree_;
};

}   // namespace fdapde

#endif   // __FDAPDE_SP_DOF_HANDLER_H__
