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

// #ifndef __FDAPDE_DCEL_H__
// #define __FDAPDE_DCEL_H__

// #include "header_check.h"
// #include <chrono>
// using namespace std::chrono;

namespace fdapde {

// implementation of the Double Connected Edge List data structure (also known as DCEL or half-edge)
template <int LocalDim, int EmbedDim> class DCEL {
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;
    using coords_t = Eigen::Matrix<double, 1, embed_dim>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    // forward decl
    struct node_t;
    struct halfedge_t;
    struct cell_t;
    // internal data structures
    struct node_t {
       private:
        int id_;                 // global node index
        halfedge_t* halfedge_;   // any edge having this node as its origin
        bool boundary_;          // asserted true if node is on boundary
        coords_t coords_;
       public:
        node_t() : id_(0), coords_(), halfedge_(nullptr), boundary_(false) { }
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
        node_t(int id, bool boundary, CoordsType&&... coords) : node_t(id, nullptr, boundary, coords...) { }

        // observers
        const coords_t& coords() const { return coords_; }
        halfedge_t* halfedge() const { return halfedge_; }
        int id() const { return id_; }
        bool on_boundary() const { return boundary_; }
        node_t* next() const { return halfedge_->next()->node(); }
        node_t* prev() const { return halfedge_->prev()->node(); }
        // modifiers
        void set_halfedge(halfedge_t* halfedge) { halfedge_ = halfedge; }
        // comparison
        bool operator==(const node_t& other) const { return id_ == other.id_; }
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
        bool on_boundary() const {
            return (
              node_->on_boundary() && twin_->node()->on_boundary() && (cell_ == nullptr || twin_->cell() == nullptr));
        }
        // modifiers
        void set_prev(halfedge_t* prev) { prev_ = prev; }
        void set_next(halfedge_t* next) { next_ = next; }
        void set_twin(halfedge_t* twin) { twin_ = twin; }
        void set_node(node_t* node) { node_ = node; }
        void set_cell(cell_t* cell) { cell_ = cell; }
        void set_id(int id) { id_ = id; }
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
        // comparison
        bool operator==(const halfedge_t& other) const { return id_ == other.id_; }
        operator bool() const { prev_ != nullptr&& next_ != nullptr; }
    };
    struct cell_t {
       private:
        int id_;
        halfedge_t* h_;
        std::vector<halfedge_t*> holes_;
       public:
        cell_t() : h_(nullptr) { }
        cell_t(int id) : id_(id), h_(nullptr) { }
        cell_t(int id, halfedge_t* h) : id_(id), h_(h) { }
        // observers
        halfedge_t* halfedge() const { return h_; }
        int id() const { return id_; }
        int n_edges() const {
            if (h_ == nullptr) return 0;
            int n = 0;
            halfedge_t* tmp = h_;
            do {
                n++;
                tmp = tmp->next();
            } while (tmp != h_);
	    return n;
        }
        bool is_triangle() const { return n_edges() == 3; }
        // modifiers
        void set_halfedge(halfedge_t* h) { h_ = h; }
        // comparison
        bool operator==(const cell_t& other) const { return id_ == other.id_; }
    };
    using halfedge_iterator = std::list<halfedge_t>::iterator;
    using const_halfedge_iterator = std::list<halfedge_t>::const_iterator;
    using node_iterator = std::list<node_t>::iterator;
    using const_node_iterator = std::list<node_t>::const_iterator;
    using cell_iterator = std::list<cell_t>::iterator;
    using const_cell_iterator = std::list<cell_t>::const_iterator;
  
    // constructors
    DCEL() : nodes_(), halfedges_() { }
    // constructs a closed loop structure linking nodes one after the other, possibily allowing holes
    static DCEL<local_dim, embed_dim> Polygon(const matrix_t& boundary, const std::vector<matrix_t>& holes) {
        fdapde_assert(boundary.rows() > 2 && boundary.cols() == embed_dim);
        for (int i = 0; i < holes.size(); ++i) {
            fdapde_assert(
              holes[i].cols() == embed_dim && internals::polygon_in_2d_polygon(holes[i] FDAPDE_COMMA boundary));
            for (int j = i + 1; j < holes.size(); ++j) {
                fdapde_assert(!internals::polygon_in_2d_polygon(holes[i] FDAPDE_COMMA holes[j]));
            }
        }
        DCEL<local_dim, embed_dim> dcel;

        // set-up external boundary
        int n_nodes = boundary.rows();
        dcel.cells_.push_back(cell_t(0));
        dcel.cells_map_.emplace(0, std::prev(dcel.cells_.end()));
        cell_t* c = std::addressof(dcel.cells_.back());
        // nodes and halfedges for external boundary
        for (int i = 0; std::cmp_less(i, n_nodes); ++i) {
            node_t* n = dcel.insert_node(node_t(i, true, boundary.row(i)));
            halfedge_t* h = dcel.emplace_halfedge(n);
            n->set_halfedge(h);
            h->set_cell(c);
	    node_map_.emplace(i, std::prev(dcel_.nodes_.end()));
	    halfedge_map_.emplace(h->id(), std::prev(dcel_.halfedges_.end()));
        }
        c->set_halfedge(dcel.nodes_begin()->halfedge());
        // twin halfedges for external boundary
        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {
            node_t* n1 = std::addressof(*it);
            node_t* n2 = std::addressof(*((it->id() == n_nodes - 1) ? dcel.nodes_begin() : std::next(it, 1)));
            halfedge_t* h1 = n1->halfedge();
            halfedge_t* h2 = dcel.emplace_halfedge(n2);   // push twin edge
            h2->set_cell(nullptr);
            h2->set_twin(h1);
            h1->set_twin(h2);
        }
        // next, prev pointers for external boundary
        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {
            halfedge_t* h1 = it->halfedge();
            halfedge_t* h2 = ((it->id() == n_nodes - 1) ? dcel.nodes_begin() : std::next(it, 1))->halfedge();
            h1->set_next(h2);
            h2->set_prev(h1);
            h1->twin()->set_prev(h2->twin());
            h2->twin()->set_next(h1->twin());
        }

        if (holes.size() > 0) {   // push holes
            int offset = dcel.n_nodes();
            for (int k = 0; std::cmp_less(k, holes.size()); ++k) {
                int n_nodes_hole = holes[k].rows();
                for (int i = 0; i < n_nodes_hole; ++i) {
                    node_t* n = dcel.insert_node(node_t(offset + i, true, holes[h].row(i)));
                    halfedge_t* h = dcel.emplace_halfedge(n); // ------------------------- rename in insert_halfedge, and provide better utilities
                    n->set_halfedge(h);
                    h->set_cell(c);
                    node_map_.emplace(i, std::prev(dcel_.nodes_.end()));
                    halfedge_map_.emplace(h->id(), std::prev(dcel_.halfedges_.end()));
                }
                for (int i = 0; i < n_nodes_hole; ++i) {
                    node_t* n1 = std::addressof(*(std::next(dcel.nodes_begin(), offset + i)));
                    node_t* n2 = std::addressof(*(std::next(dcel.nodes_begin(), offset + (i + 1) % n_nodes_hole)));
                    halfedge_t* h1 = n1->halfedge();
                    halfedge_t* h2 = dcel.emplace_halfedge(n2);
                    h1->set_twin(h2);
                    h2->set_twin(h1);
                    h2->set_cell(nullptr);   // set h2 cell to nullptr, as it is on boundary
                }
                // connect hole's nodes
                for (int i = 0; i < hole_nodes; ++i) {
                    halfedge_t* h1 = std::next(dcel.nodes_begin(), offset + i)->halfedge();
                    halfedge_t* h2 = std::next(dcel.nodes_begin(), offset + (i + 1) % n_nodes_hole)->halfedge();
                    h1->set_next(h2);
                    h2->set_prev(h1);
                    h1->twin()->set_prev(h2->twin());
                    h2->twin()->set_next(h1->twin());
                }
                offset += n_nodes_hole;
            }
        }
        return dcel;
    }
    static DCEL<local_dim, embed_dim> Polygon(const matrix_t& boundary) {
        return DCEL<local_dim, embed_dim>::Polygon(boundary, std::vector<matrix_t> {});
    }
    // observers
    matrix_t nodes() const {   // matrix of nodes coordinates
        matrix_t coords(nodes_.size(), embed_dim);
        for (int i = 0, n = nodes_.size(); i < n; ++i) { coords.row(nodes[i].id()) = nodes[i].coords(); }
        return coords;
    }
    int n_nodes() const { return nodes_.size(); }
    int n_halfedges() const { return halfedges_.size(); }   
    int n_cells() const { return cells_.size(); }
    int n_edges() const { return halfedges_ .size()/ 2; }

  // bool is_triangulation() const {}
  
    // iterators
    // cyclic iteration over half-edge chain
    typename halfedge_t::circulator halfedge_circulator(halfedge_t* halfedge) {
        return typename halfedge_t::circulator(halfedge);
    }
    halfedge_iterator halfedges_begin() { return halfedges_.begin(); }
    halfedge_iterator halfedges_end() { return halfedges_.end(); }
    const_halfedge_iterator halfedges_cbegin() const { return halfedges_.cbegin(); }
    const_halfedge_iterator halfedges_cend() const{ return halfedges_.cend(); }
    node_iterator nodes_begin() { return nodes_.begin(); }
    node_iterator nodes_end() { return nodes_.end(); }
    const_node_iterator nodes_cbegin() const { return nodes_.cbegin(); }
    const_node_iterator nodes_cend() const{ return nodes_.cend(); }
    cell_iterator cells_begin() { return cells_.begin(); }
    cell_iterator cells_end() { return cells_.end(); }
    const_cell_iterator cells_cbegin() const{ return cells_.cbegin(); }
    const_cell_iterator cells_cend() const{ return cells_.cend(); }

    // modifiers
    node_t* insert_node(const node_t& node) {
        nodes_.push_back(node);
        return std::addressof(nodes_.back());
    }
    // insert edge between halfedges v1 and v2
    halfedge_t* insert_edge(halfedge_t* v1, halfedge_t* v2) {
        if (
          (v1 == nullptr || v2 == nullptr) || (v1 == v2) ||                           // nothing to do
          (v1->node() == v2->next()->node() || v2->node() == v1->next()->node())) {   // v1 and v2 are next halfedges
            return v1;
        }
        // get nodes from v1 and v2
        node_t* n1 = v1->node();
        node_t* n2 = v2->node();
        // create a pair of twin half-edges (if any between v1 and v2 is unbounded, use it)
        halfedge_t* h1 = bool(*v1) ? emplace_halfedge(n1) : v1;
        halfedge_t* h2 = bool(*v2) ? emplace_halfedge(n2) : v2;
        h1->set_twin(h2);
        h2->set_twin(h1);
        // insert halfedge h1 between v1 and v1->prev
        h2->set_next(v1);
        if (v1->prev()) {
            v1->prev()->set_next(h2->twin());
            h2->twin()->set_prev(v1->prev());
            v1->set_prev(h2);
        } else {   // make a closed half-loop
            h1->set_prev(h2);
            h2->set_next(h1);
        }
        h2->next()->set_prev(h2);
        h2->set_node(n2);
        // insert halfedge h2 between v2 and v2->prev
        h1->set_next(v2);
        if (v2->prev()) {
            v2->prev()->set_next(h1->twin());
            h1->twin()->set_prev(v2->prev());
            v2->set_prev(h1);
        } else {   // make a closed half-loop
            h2->set_prev(h1);
            h1->set_next(h2);
        }
        h1->next()->set_prev(h1);
        h1->set_node(n1);

        // set cell pointers
        halfedge_t* tmp = h1;
        do { tmp = tmp->next(); } while (tmp != h1 && tmp != h2);
        if (tmp == h1) {   // h2 not on the same chain of h1. edge insertion splitted the current cell in two
            h1->set_cell(h1->prev()->cell());
            h1->cell()->set_halfedge(h1);
	    // create new cell
            int id = std::prev(cells_.end())->id() + 1;
            cells_.push_back(cell_t(id));
            cell_t* c = std::addressof(cells_.back());
            c->set_halfedge(h2);
	    cells_map_.emplace(id, std::prev(cells_.end()));
	    // update all h2's chain to point to new created cell
            halfedge_t* end = h2;
            do {
                h2->set_cell(c);
                h2 = h2->next();
            } while (h2 != end);
        } else {   // degenerate case: h1 and h2 are on the same loop
            if (!(h1->next() == h2 && h2->next() == h1)) {
                // still h1 and h2 do not form a closed isolated edge
                h1->set_cell(h1->next() != h2 ? h1->next()->cell() : h1->prev()->cell());
                h2->set_cell(h1->cell());
            }
        }
        return h1;
    }
    // remove edge of which v is one of its halfedges
    halfedge_t* remove_edge(halfedge_t* v){
        fdapde_assert(v != nullptr);

	halfedge_t* v1 = v1->cell() == nullptr ? v->twin() : v; // take v's twin if v external halfedge on boundary 
        halfedge_t* v2 = v1->twin();
	
        // set next of v1_prev to v1_next
        v1->prev()->set_next(v1->next());      
        v2->next()->set_prev(v1->prev());
        // set next of v2_prev to v1_next
        v2->prev()->set_next(v1->next());
        v1->next()->set_prev(v2->prev());

	
        halfedge_t* end;
        halfedge_t* begin;
        cell_t* c1 = v1->cell();

        // if v1 is not a boundary edge, assign halfedges of v2's cell to v1's cell (collapse 2 cells into 1)
        if(!v1->on_boundary()){
            begin = v2->next();
            end = v2;
            c1->set_halfedge(begin); 
            do {
                begin->set_cell(c1);
                begin = begin->next();
            } while (begin != end);
        }
        // if v1 is a boundary edge, set the halfedges in v1's cell to boundary halfedges, cell=null and remove v1's cell
        else{
            begin= v1->next();
            end = v1;
            do {
                begin->set_cell(nullptr);
                begin->node()->set_boundary(true);
                begin->twin()->node()->set_boundary(true);
                begin = begin->next();
            } while (begin != end);
            // remove v1's cell
            cells_.erase(c1->it());  
        }


        // remove v2's cell
        cell_t* c2=v2->cell();
        if (c2) {
            cells_.erase(c2->it());
        }
        // remove v1 and v2
        halfedge_t* next= v2->next();
        halfedges_.erase(v1->it());  
        halfedges_.erase(v2->it());
        return next; 
    }
  
    // insert a cell (ordered list of nodes) attached by v to the DCEL
    // halfedge_t* add_cell(halfedge_t* v, const std::vector<node_t*>& nodes){
    //     int nodes_polygon= nodes.size();                                                                         
    //     cell_t* c= v->cell();                                                                                    
    
    //     std::vector<halfedge_t*> halfedges_to_call(nodes_polygon +2 ); 
    //     halfedges_to_call[0] = v;
    //     halfedges_to_call[1] = v->next();
    
    //     // fill halfedges_to_call in order to make the right calls  to insert_edge
    //     for (int i = 0; i < nodes_polygon; ++i) {
    //         halfedge_t* h = find_halfedge(nodes[i],c, building_dcel);
    //         if(!h){ //halfedge doesn't exist yet
    //                 h = emplace_halfedge(nodes[i]);
    //                 h->set_cell(c);
    //         }
    //         halfedges_to_call[i+2] = h;
    //     }
    //     // add edges
    //     for (int i = 0; i < nodes_polygon+2 ; ++i) {
    //         halfedge_t* h1 = halfedges_to_call[i];
    //         halfedge_t* h2 = halfedges_to_call[(i + 1) % (nodes_polygon + 2)];
    //         halfedges_to_call[(i + 1) % (nodes_polygon + 2)] = insert_edge(h1, h2)->next();
    //     }
    //     c->set_halfedge(v);
    //     return v;
    // }

    // // remove polygon inside dcel by calling its cell's pointer
    // void remove_polygon(cell_t* cell) {
    //     if(!cell) return; // nothing to remove
    //     // removing cell's halfedges
    //     halfedge_t* h1 = cell->halfedge();
    //     halfedge_t* ending = h1->twin() ? h1->twin()->next() : nullptr; 
    //     do {
    //         halfedge_t* next = remove_edge(h1->twin());
    //         h1 = next; 
    //     } while (h1 && h1 != ending);
    // }   

    // return the node of the halfedge previous to h 
  node_t* adjacent(halfedge_t* h) const {return (h->twin()) ? h->twin()->prev()->node() : nullptr;  } // MUST BE A METHOD OF HALFEDGE THEN!!!!


    // find the halfedge given its node and cell
  halfedge_t* find_halfedge(node_t* n, cell_t* cell, bool building_dcel=false) { // WITH THE MAPS, SHOULD BE POSSIBLE IN O(1)
        if(!building_dcel){
            halfedge_t* h= cell->halfedge();
            halfedge_t* end=h;
            do{
                if(h->node()==n)
                    return h;
                h = h->next();
            }while(h!=end);
        }
        // if the DCEL is being built from scratch, we need to iterate over all halfedges, since holes' halfedges might not be connected to the rest of the DCEL yet
        else{ 
            for(auto it = halfedges_begin(); it != halfedges_end(); ++it) {
                halfedge_t* h = &(*it);
                if (h->node() == n && h->cell() == cell) {
                    return h;
                }
            }
        }
        return nullptr;
    }

    halfedge_t* find_halfedge_between(coords_t from, coords_t to) {
        for (auto it = halfedges_begin(); it != halfedges_end(); ++it) {
            halfedge_t* h = &(*it);
            if (h->node()->coords() == from && h->twin() && h->twin()->node()->coords() == to) {
                return h;
            }
        }
        return nullptr;
    }

    node_t* find_node(const coords_t& coords) {
        for (auto it = nodes_begin(); it != nodes_end(); ++it) {
            if (it->coords() == coords) {
                return std::addressof(*it);
            }
        }
        return nullptr;
    }
    
  
    // utils
    template <typename... Args> halfedge_t* emplace_halfedge(Args&&... args) {
        int id= std::prev(halfedges_end())->id() + 1; // get the last halfedge id and increment it
        halfedges_.emplace_back(id++, std::forward<Args>(args)...);
        auto it = std::prev(halfedges_.end()); 
        it->set_it(it); // = it;
        return std::addressof(halfedges_.back());
    }

    template <typename... Args> node_t* emplace_node(Args&&... args) {
        int id= std::prev(nodes_end())->id() + 1;
        nodes_.emplace_back(id++, std::forward<Args>(args)...);
        return std::addressof(nodes_.back());
    }  

    // converts a DCEL object into a TriangulationType object
    template <typename TriangulationType>
    TriangulationType to_triangulation() const {
        Eigen::Matrix<double, Eigen::Dynamic, embed_dim> nodes_mat(n_nodes(), embed_dim);
        Eigen::Matrix<int, Eigen::Dynamic, 3> cells_mat(n_cells(), 3);
        Eigen::Matrix<int, Eigen::Dynamic, 1> boundary_markers(n_nodes());

        int idx = 0;
        for (auto it = nodes_cbegin(); it != nodes_cend(); ++it, ++idx) {
            nodes_mat.row(it->id()) = it->coords().transpose();
            boundary_markers(it->id()) = it->on_boundary() ? 1 : 0;
        }

        idx = 0;
        for (auto it = cells_cbegin(); it != cells_cend(); ++it, ++idx) {
            halfedge_t* h = it->halfedge();
            cells_mat(idx, 0) = h->node()->id();
            cells_mat(idx, 1) = h->next()->node()->id();
            cells_mat(idx, 2) = h->prev()->node()->id();
        }
        return TriangulationType(nodes_mat, cells_mat, boundary_markers);
    }

    // converts a TriangulationType object into a DCEL object, knowing its holes
    template <typename TriangulationType>
    void from_triangulation(const TriangulationType& triangulation, const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& holes) {
        int n_hole_nodes = 0;
        for (const auto& hole : holes)
            n_hole_nodes += hole.rows();
        int n_boundary_external = triangulation.n_boundary_nodes() - n_hole_nodes;
        Eigen::Matrix<double, Eigen::Dynamic, embed_dim> boundary_nodes(n_boundary_external, embed_dim);
        const auto& coords = triangulation.nodes();
        const auto& markers = triangulation.boundary_nodes();
        int n = boundary_nodes.rows();
        
        /*int mid = n / 2;  // punto di taglio (metà inferiore se dispari)
        // Prima metà
        Eigen::Matrix<double, Eigen::Dynamic, embed_dim> part1 = boundary_nodes.topRows(mid);
        // Seconda metà
        Eigen::Matrix<double, Eigen::Dynamic, embed_dim> part2 = boundary_nodes.bottomRows(n - mid);*/
        
        int idx = 0;
        for (int i = 0; i < n_boundary_external; ++i) {
            if (markers(i, 0) == 1) {
                boundary_nodes.row(idx++) = coords.row(i);
            }
        }
        // build the DCEL fron scratch
        // otherwise, if the DCEL is not empty, add to the pre-existing structure
        if(this->nodes_.empty() && this->halfedges_.empty() && this->cells_.empty()) { 
            *this = DCEL::make_polygon(boundary_nodes, holes);
        }

        //add internal nodes
        for (int i = 0; i < coords.rows(); ++i) {
            if (markers(i, 0) == 0) {
                insert_node(typename DCEL::node_t(n_nodes(), false, coords.row(i)));
            }
        }

        auto cells= triangulation.cells();

        // creating a map to associate nodes to holes
        std::unordered_map<int, int> node_to_hole;
        std::vector<std::set<std::pair<int, int>>> hole_edges(holes.size());
        for (int k = 0; k < holes.size(); ++k) {
            const auto& hole = holes[k];
            std::vector<int> hole_node_ids;  
            for (int i = 0; i < hole.rows(); ++i) {
                const auto& pt = hole.row(i);
                for (int j = 0; j < triangulation.nodes().rows(); ++j) {
                    if ((triangulation.nodes().row(j) - pt).norm() < 1e-10) {
                        node_to_hole[j] = k;
                        hole_node_ids.push_back(j);
                        break;
                    }
                }
            }
            int m = hole_node_ids.size();
            for (int i = 0; i < m; ++i) {
                int a = hole_node_ids[i];
                int b = hole_node_ids[(i + 1) % m];  
                hole_edges[k].insert({a, b});  
            }
        }
        // vector of bools to understand if an edge of a hole has been connected to the rest of the dcel being created
        std::vector<bool> is_connected(holes.size(),false);
        cell_t* longest_cell = nullptr;
        std::list< Eigen::Matrix<int, Eigen::Dynamic, 3> > cells_list;
        for(int i = 0; i < triangulation.n_cells(); ++i) {
            cells_list.push_back(Eigen::Matrix<int, Eigen::Dynamic, 3>(triangulation.cells().row(i)));
        }
        
        // add edges and cells
        while(!cells_list.empty()) {

            auto cell = cells_list.front();
            cells_list.pop_front();
            
            int id0 = cell(0, 0);
            int id1 = cell(0, 1);
            int id2 = cell(0, 2);

            // first, for each hole connect as first a triangle that has an edge on the hole
            // to better deal with the new cells being created (so that first cells are either these triangles or the larger cells encapsulating the rest of the boundary) 
            bool all_connected = std::all_of(is_connected.begin(), is_connected.end(), [](bool v) { return v; });
            if (!all_connected){
                // map node -> hole (-1 if external)
                int h0 = node_to_hole.count(id0) ? node_to_hole.at(id0) : -1;
                int h1 = node_to_hole.count(id1) ? node_to_hole.at(id1) : -1;
                int h2 = node_to_hole.count(id2) ? node_to_hole.at(id2) : -1;

                // verify if it's a priority triangle (one that connects the hole to the boundary)
                bool is_priority = false;
                int hole_to_connect = -1;

                // to check if 2 nodes in same hole are actually consecutive (on same edge)
                auto is_consecutive_in_hole = [&](int a, int b, int hole_id) {
                    return hole_edges[hole_id].count({a, b}) || hole_edges[hole_id].count({b, a});
                };
                if (h0 == h1 && h0 != -1 && h2 == -1 && !is_connected[h0] && is_consecutive_in_hole(id0, id1, h0)) {
                    is_priority = true;
                    hole_to_connect = h0;
                }
                else if (h1 == h2 && h1 != -1 && h0 == -1 && !is_connected[h1] && is_consecutive_in_hole(id1, id2, h1)) {
                    is_priority = true;
                    hole_to_connect = h1;
                }
                else if (h2 == h0 && h2 != -1 && h1 == -1 && !is_connected[h2] && is_consecutive_in_hole(id2, id0, h2)) {
                    is_priority = true;
                    hole_to_connect = h2;
                }
                
                if (!is_priority) {
                    // go to the next cell
                    cells_list.push_back(cell);
                    continue;
                }
                is_connected[hole_to_connect] = true;
            }

            int row0 = cell(0, 0);
            int row1 = cell(0, 1);
            int row2 = cell(0, 2);
            node_t* p0 = find_node(coords.row(row0));
            node_t* p1 = find_node(coords.row(row1));
            node_t* p2 = find_node(coords.row(row2));

            if (!fdapde::internals::are_2d_counterclockwise_sorted(p0->coords(), p1->coords(), p2->coords())) {
                std::swap(p1, p2);
            }

            halfedge_t* h = find_halfedge_between(p0->coords(), p1->coords());
            bool building_dcel = true; 
            if (h) {
                add_polygon(h, {p2}, building_dcel);
            } else if ((h = find_halfedge_between(p1->coords(), p2->coords()))) {
                add_polygon(h, {p0}, building_dcel);
            } else if ((h = find_halfedge_between(p2->coords(), p0->coords()))) {
                add_polygon(h, {p1}, building_dcel);
            } else {
                // there might be 2 or more halfedges departing from same node and belonging to same cell
                // e.g. if each of the three edges of the triangle is adjacent to a different region:
                // one to the external boundary, one to an internal hole, and the third to another distinct internal hole.
                cells_list.push_back(cell);
                continue;
            }
              
            if(!all_connected){
                // find the cell that connects all the halfedges that don't belong to triangles yet 
                // either the cell of the last halfedge created or its twin's
                halfedge_t* h_last= &(*std::prev(halfedges_end()));  
                int cont=1;
                halfedge_t* h= h_last->next();
                do{
                    cont++;
                    h=h->next();
                }while(h!=h_last);
                if(cont>3){  // cell is not a triangle
                    longest_cell= h_last->cell();
                }
                else if(cont==3)
                    longest_cell= h_last->twin()->cell();

                //assign the cell to the halfedges of the holes that are not connected yet
                for(int k = 0; k < holes.size(); ++k) {
                    if(!is_connected[k]){  
                        coords_t co1= holes[k].row(0);
                        coords_t co2= holes[k].row(1);
                        halfedge_t* h_hole = find_halfedge_between(co1, co2);  //holes[k] is clockwise sorted
                        h_hole->set_cell(longest_cell);  // set the cell of the halfedge to the longest cell
                        halfedge_t* h_next = h_hole->next();
                        do{
                            h_next->set_cell(longest_cell);  
                            h_next = h_next->next();
                        }while(h_next != h_hole);
                    }
                }
            }  
        }
    }
   private:
    // internal storage (use list to avoid reallocations)
    std::list<halfedge_t> halfedges_;
    std::list<cell_t> cells_;
    std::list<node_t> nodes_;
    // fast indexing
    std::unordered_map<int, std::list<halfedge_t>::iterator> halfedges_map_;
    std::unordered_map<int, std::list<cell_t>::iterator> cells_map_;
    std::unordered_map<int, std::list<node_t>::iterator> nodes_map_;
};

}   // namespace fdapde

#endif // __DCEL_H__
