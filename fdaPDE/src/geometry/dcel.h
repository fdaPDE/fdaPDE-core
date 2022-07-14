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

#include "header_check.h"

namespace fdapde {
  
// implementation of the Double Connected Edge List data structure (also known as DCEL or half-edge)
template <int LocalDim, int EmbedDim> class DCEL {
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;
    // forward decl
    struct node_t;
    struct halfedge_t;
    struct cell_t;

    // internal data structures
    struct node_t {
       private:
        using coords_t = Eigen::Matrix<double, embed_dim, 1>;
        int id_;                  // global node index
        halfedge_t* halfedge_;    // any edge having this node as its origin
        bool boundary_;           // asserted true if node is on boundary
        coords_t coords_;
       public:
        node_t() : coords_(), halfedge_(nullptr), boundary_(false) { }
        template <typename CoordsType>
            requires(internals::is_eigen_dense_xpr_v<CoordsType>)
        node_t(int id, halfedge_t* halfedge, bool boundary, const CoordsType& coords) :
            id_(id), halfedge_(halfedge), boundary_(boundary), coords_() {
            fdapde_assert(
              (coords.rows() == 1 && coords.cols() == embed_dim) || (coords.rows() == embed_dim && coords.cols() == 1));
            if (coords.rows() == 1) {
                coords_ = coords.transpose();
            } else {
                coords_ = coords;
            }
        }
        template <typename CoordsType>
            requires(internals::is_eigen_dense_xpr_v<CoordsType>)
        node_t(int id, bool boundary, const CoordsType& coords) : node_t(id, nullptr, boundary, coords) { }
        template <typename... CoordsType>
            requires(std::is_floating_point_v<CoordsType> && ...) && (sizeof...(CoordsType) == embed_dim)
        node_t(int id, halfedge_t* halfedge, bool boundary, CoordsType&&... coords) :
            id_(id), halfedge_(halfedge), boundary_(boundary), coords_(coords...) { }
        template <typename... CoordsType>
            requires(std::is_floating_point_v<CoordsType> && ...) && (sizeof...(CoordsType) == embed_dim)
        node_t(int id, bool boundary, CoordsType&&... coords) :
            node_t(id, nullptr, boundary, coords...) { }
        // observers
        const Eigen::Matrix<double, embed_dim, 1>& coords() const { return coords_; }
        halfedge_t* halfedge() const { return halfedge_; }
        void set_halfedge(halfedge_t* halfedge) { halfedge_ = halfedge; }
        int id() const { return id_; }
        bool on_boundary() const { return boundary_; }
        node_t* next() const { return halfedge_->next()->node(); }
        node_t* prev() const { return halfedge_->prev()->node(); }
    };
    struct halfedge_t {
       private:
        int id_;   // global halfedge index
        halfedge_t *prev_, *next_, *twin_;
        node_t* node_;
        cell_t* cell_;   // cell to which this halfedge belongs to
       public:
        halfedge_t() : node_(nullptr), prev_(nullptr), next_(nullptr), twin_(nullptr) { }
        halfedge_t(int id, halfedge_t* prev, halfedge_t* next, halfedge_t* twin, node_t* node) :
            id_(id), prev_(prev), next_(next), twin_(twin), node_(node) { }
        // no twin constructors
        halfedge_t(int id, halfedge_t* prev, halfedge_t* next, node_t* node) :
            halfedge_t(id, prev, next, nullptr, node) { }
        // minimal constructor
        halfedge_t(int id, node_t* node) : halfedge_t(id, nullptr, nullptr, nullptr, node) { }

        // observers
        halfedge_t* prev() const { return prev_; }
        halfedge_t* next() const { return next_; }
        halfedge_t* twin() const { return twin_; }
        node_t* node() const { return node_; }
        cell_t* cell() const { return cell_; }
        int id() const { return id_; }
        bool on_boundary() const { return node_->on_boundary() && twin_->node()->on_boundary(); }
        // modifiers
        void set_prev(halfedge_t* prev) { prev_ = prev; }
        void set_next(halfedge_t* next) { next_ = next; }
        void set_twin(halfedge_t* twin) { twin_ = twin; }
        void set_node(node_t* node) { node_ = node; }
        void set_cell(cell_t* cell) { cell_ = cell; }

        // iterator (follows the chain of directed edges until no next valid edge or this edge is found)
        struct circulator {
            using value_type = halfedge_t;
            using pointer = std::add_pointer_t<value_type>;
            using reference = std::add_lvalue_reference_t<value_type>;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::forward_iterator_tag;

            circulator(halfedge_t* halfedge) :
                halfedge_(halfedge), end_(halfedge == nullptr ? nullptr : halfedge->prev()) { }
            circulator& operator++() {
                if (last_) { [[unlikely]]
                    end_ = nullptr;
                } else {
                    halfedge_ = halfedge_->next();
                    if (halfedge_ == end_) { last_ = true; }   // implement cyclic structure
                }
                return *this;
            }
            // access
            pointer operator->() { return halfedge_; }
            const pointer operator->() const { return halfedge_; }
            reference operator*() { return *halfedge_; }
            const reference operator*() const { return *halfedge_; }
            operator bool() const { return end_ == nullptr; }
            // comparison
            friend bool operator==(const circulator& lhs, const circulator& rhs) { return lhs.end_ == rhs.end_; }
            friend bool operator!=(const circulator& lhs, const circulator& rhs) { return lhs.end_ != rhs.end_; }
           private:
            bool last_ = false;
            pointer halfedge_, end_;
        };
    };
    struct cell_t {
        cell_t() : h_(nullptr) { }
        cell_t(int id) : id_(id), h_(nullptr) { }
        cell_t(int id, halfedge_t* h) : id_(id), h_(h) { }
        // observers
        halfedge_t* halfedge() const { return h_; }
        int id() const { return id_; }
        // modifiers
        void set_halfedge(halfedge_t* h) { h_ = h; }
       private:
        int id_;
        halfedge_t* h_;
    };
    using halfedge_iterator = std::list<halfedge_t>::iterator;
    using node_iterator = std::list<node_t>::iterator;
    using cell_iterator = std::list<cell_t>::iterator;

    // constructors
    DCEL() : nodes_(), halfedges_(), n_nodes_(0), n_halfedges_(0), n_cells_(0) { }
    // constructs a closed loop structure linking nodes one after the other
    static DCEL<local_dim, embed_dim> make_polygon(const Eigen::Matrix<double, Dynamic, Dynamic>& nodes) {
        fdapde_assert(nodes.cols() == embed_dim);
        int n_nodes = nodes.rows();
        int n_halfedges = 2 * (n_nodes + 1);
        DCEL<local_dim, embed_dim> dcel;
        // create polygon cell
        dcel.cells_.push_back(cell_t(0));
        cell_t* c = std::addressof(dcel.cells_.back());
        dcel.n_cells_ = 1;
        // push nodes
        for (int i = 0; i < n_nodes; ++i) {
            node_t* n = dcel.insert_node(node_t(i, /* boundary = */ true, nodes.row(i)));
            halfedge_t* h = dcel.emplace_halfedge_(n);
            n->set_halfedge(h);
	    h->set_cell(c);
        }
        // create twin edges
        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {
            node_t* n1 = std::addressof(*it);
            node_t* n2 = std::addressof(*((it->id() == n_nodes - 1) ? dcel.nodes_begin() : std::next(it, 1)));
            halfedge_t* h1 = n1->halfedge();
            halfedge_t* h2 = dcel.emplace_halfedge_(n2);   // push twin edge
            h2->set_twin(h1);
            h1->set_twin(h2);
        }
        // finalize next-prev pointer pairs
        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {
            halfedge_t* h1 = it->halfedge();
            halfedge_t* h2 = ((it->id() == n_nodes - 1) ? dcel.nodes_begin() : std::next(it, 1))->halfedge();
            h1->set_next(h2);
            h2->set_prev(h1);
	    h1->twin()->set_prev(h2->twin());
            h2->twin()->set_next(h1->twin());
        }
        return dcel;
    }
    // modifiers
    node_t* insert_node(const node_t& node) {
        nodes_.push_back(node);
	n_nodes_++;
        return std::addressof(nodes_.back());
    }
    halfedge_t* insert_edge(halfedge_t* v1, halfedge_t* v2) {
        if (v1->node() == v2->next()->node()) return v1;   // v1 and v2 are next halfedges
        // get exiting halfedges from n1 and n2
        node_t* n1 = v1->node();
        node_t* n2 = v2->node();
        // create a pair of twin half-edges
        halfedge_t* h1 = emplace_halfedge_(n1);
        halfedge_t* h2 = emplace_halfedge_(n2);
        h1->set_twin(h2);
        h2->set_twin(h1);

	// insert halfedge h1 between v1 and v1->prev
        h2->set_next(v1);
        v1->prev()->set_next(h2->twin());
	h2->twin()->set_prev(v1->prev());
        h2->next()->set_prev(h2);
        h2->set_node(n2);
	// insert halfedge h2 between v2 and v2->prev
        h1->set_next(v2);
        v2->prev()->set_next(h1->twin());
	h1->twin()->set_prev(v2->prev());
        h1->next()->set_prev(h1);
        h1->set_node(n1);

	// set cell pointers
        h1->set_cell(v1->cell());
        h1->cell()->set_halfedge(h1);
        // create new cell
        cells_.push_back(cell_t(n_cells_++));
        cell_t* c1 = std::addressof(cells_.back());
        c1->set_halfedge(h2);
        halfedge_t* end = h2;
        do {
            h2->set_cell(c1);
            h2 = h2->next();
        } while (h2 != end);
	return h1;
    }

    // observers
    Eigen::Matrix<double, Dynamic, Dynamic> nodes() const {   // matrix of nodes coordinates
        Eigen::Matrix<double, Dynamic, Dynamic> coords(n_nodes_, embed_dim);
        for (int i = 0; i < n_nodes_; ++i) { coords.row(nodes[i].id()) = nodes[i].coords(); }
        return coords;
    }
    int n_nodes() const { return n_nodes_; }
    int n_halfedges() const { return n_halfedges_; }
    int n_cells() const { return n_cells_; }
    int n_edges() const { return n_halfedges_ / 2; }

    // iterators
    // cyclic iteration over half-edge chain
    typename halfedge_t::circulator halfedge_circulator(halfedge_t* halfedge) {
        return typename halfedge_t::circulator(halfedge);
    }
    halfedge_iterator halfedges_begin() { return halfedges_.begin(); }
    halfedge_iterator halfedges_end() { return halfedges_.end(); }
    node_iterator nodes_begin() { return nodes_.begin(); }
    node_iterator nodes_end() { return nodes_.end(); }
    cell_iterator cells_begin() { return cells_.begin(); }
    cell_iterator cells_end() { return cells_.end(); }
  
   private:
    // internal utils
    template <typename... Args> halfedge_t* emplace_halfedge_(Args&&... args) {
        halfedges_.emplace_back(n_halfedges_++, std::forward<Args>(args)...);
        return std::addressof(halfedges_.back());
    }
    template <typename... Args> node_t* emplace_node_(Args&&... args) {
        nodes_.emplace_back(n_nodes_++, std::forward<Args>(args)...);
        return std::addressof(nodes_.back());
    }  
    // internal storage (use list to avoid reallocations)
    std::list<node_t> nodes_;
    std::list<halfedge_t> halfedges_;
    std::list<cell_t> cells_;
    int n_nodes_, n_halfedges_, n_cells_;
};
  
}   // namespace fdapde

#endif // __DCEL_H__
