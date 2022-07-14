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

#ifndef __FDAPDE_POLYGON_H__
#define __FDAPDE_POLYGON_H__

#include "header_check.h"

namespace fdapde {

// a polygon is a connected list of vertices, possibly formed by more than one ring
template <int LocalDim, int EmbedDim> class Polygon {
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;

    // constructors
    Polygon() noexcept = default;
    Polygon(const Eigen::Matrix<double, Dynamic, Dynamic>& nodes) noexcept : triangulation_() {
        fdapde_assert(nodes.rows() > 0 && nodes.cols() == embed_dim);
        if (internals::are_2d_counterclockwise_sorted(nodes)) {
            triangulate_(nodes);
        } else {   // nodes are in clocwise order, reverse node ordering
            int n_nodes = nodes.rows();
            Eigen::Matrix<double, Dynamic, Dynamic> reversed_nodes(n_nodes, embed_dim);
            for (int i = 0; i < n_nodes; ++i) { reversed_nodes.row(i) = nodes.row(n_nodes - 1 - i); }
            triangulate_(reversed_nodes);
        }
    }
  
    Polygon(const Polygon&) noexcept = default;
    Polygon(Polygon&&) noexcept = default;  
    // observers
    const Eigen::Matrix<double, Dynamic, Dynamic>& nodes() const { return triangulation_.nodes(); }
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& cells() const { return triangulation_.cells(); }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> edges() const {
        return triangulation_.edges();
    }
    const Triangulation<local_dim, embed_dim>& triangulation() const { return triangulation_; }
    double measure() const { return triangulation_.measure(); }
    int n_nodes() const { return triangulation_.n_nodes(); }
    int n_cells() const { return triangulation_.n_cells(); }
    int n_edges() const { return triangulation_.n_edges(); }
    // test whether point p is contained in polygon
    template <int Rows, int Cols>
        requires((Rows == embed_dim && Cols == 1) || (Cols == embed_dim && Rows == 1))
    bool contains(const Eigen::Matrix<double, Rows, Cols>& p) const {
        return triangulation_.locate(p) != -1;
    }
    // random sample points in polygon
    Eigen::Matrix<double, Dynamic, Dynamic> sample(int n_samples, int seed = random_seed) const {
        return triangulation_.sample(n_samples, seed);
    }
   private:
    // perform polygon triangulation
    void triangulate_(const Eigen::Matrix<double, Dynamic, Dynamic>& nodes) {
        std::vector<int> cells;
        // perform monotone partitioning
        std::vector<std::vector<int>> poly_partition = monotone_partition_(nodes);
        // triangulate each monotone polygon
        for (const std::vector<int>& poly : poly_partition) {
            std::vector<int> local_cells = triangulate_monotone_(nodes(poly, Eigen::all));
            // move local node numbering to global node numbering
            for (std::size_t i = 0; i < local_cells.size(); ++i) { local_cells[i] = poly[local_cells[i]]; }
            cells.insert(cells.end(), local_cells.begin(), local_cells.end());
        }
        // set-up face-based data structure
        triangulation_ = Triangulation<LocalDim, EmbedDim>(
          nodes, Eigen::Map<Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(cells.data(), cells.size() / 3, 3),
          Eigen::Matrix<int, Dynamic, 1>::Ones(nodes.rows()));
    }

    // partition an arbitrary polygon P into a set of monotone polygons (plane sweep approach, section 3.2 of De Berg,
    // M. (2000). Computational geometry: algorithms and applications. Springer Science & Business Media.)
    std::vector<std::vector<int>> monotone_partition_(const Eigen::Matrix<double, Dynamic, Dynamic>& coords) {      
        using poly_t = DCEL<local_dim, embed_dim>;
	using halfedge_t = typename poly_t::halfedge_t;
	using halfedge_ptr_t = std::add_pointer_t<halfedge_t>;
        // data structure inducing an ad-hoc edge ordering for monotone partitioning
        struct edge_t {
            double p1x, p1y;   // p1 coordinates
            double p2x, p2y;   // p2 coordinates
            int id;
            // constructor
            edge_t(double p1x_, double p1y_, double p2x_, double p2y_) noexcept :
                p1x(p1x_), p1y(p1y_), p2x(p2x_), p2y(p2y_) { }
            edge_t(int id_, double p1x_, double p1y_, double p2x_, double p2y_) noexcept :
                p1x(p1x_), p1y(p1y_), p2x(p2x_), p2y(p2y_), id(id_) { }
            // right-to-left edge ordering relation along x-coordinate
            bool operator<(const edge_t& rhs) const {   // for monotone partitioning, rhs is always below the p1 point
                if (rhs.p1y == rhs.p2y) {                // rhs is horizontal
                    if (p1y == p2y) { return (p1y < rhs.p1y); }   // both edges are horizontal lines
                    return internals::orientation(
                             std::array {p2x, p2y}, std::array {p1x, p1y}, std::array {rhs.p1x, rhs.p1y}) ==
                           internals::Orientation::LEFT;
                } else if (p1y == p2y || p1y < rhs.p1y) {
                    return internals::orientation(
                             std::array {rhs.p2x, rhs.p2y}, std::array {rhs.p1x, rhs.p1y}, std::array {p1x, p1y}) !=
                           internals::Orientation::LEFT;
                } else {
                    return internals::orientation(
                             std::array {p2x, p2y}, std::array {p1x, p1y}, std::array {rhs.p1x, rhs.p1y}) ==
                           internals::Orientation::LEFT;
                }
            }
        };
	// given nodes p1, p2, asserts true if p1 is below p2 (induces a y-decresing ordering on polygon nodes)
	auto below = []<typename node_t>(const node_t& p1, const node_t& p2) {
            if (p1[1] < p2[1]) {
                return true;
            } else if (p1[1] == p2[1]) {
                if (p1[0] < p2[0]) { return true; }
            }
            return false;	  
	};
	
        // O(n) polygon construction as Doubly Connected Edge List
        poly_t dcel = DCEL<local_dim, embed_dim>::make_polygon(coords);
        int n_nodes = dcel.n_nodes();
        int n_edges = dcel.n_edges();
        std::set<edge_t> sweep_line;   // edges pierced by sweep line, sorted by x-coord
        std::vector<halfedge_ptr_t> helper(n_edges, nullptr);
        std::vector<halfedge_ptr_t> nodes(n_nodes);   // maps node id to one of its halfedges

        // O(n) node type detection
        enum node_category_t { start = 0, split = 1, end = 2, merge = 3, regular = 4 };
        std::unordered_map<halfedge_ptr_t, node_category_t> node_category;
        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {   // counterclocwise order loop
            nodes[it->id()] = it->halfedge();
            auto prev = it->prev()->coords();
            auto curr = it->coords();
            auto next = it->next()->coords();
            double m_signed = internals::signed_measure_2d_tri(prev, curr, next);
            if (m_signed >= 0) {   // interior angle less or equal than \pi
                if (below(prev, curr) && below(next, curr)) {
                    node_category[nodes[it->id()]] = node_category_t::start;
                } else if (below(curr, prev) && below(curr, next)) {
                    node_category[nodes[it->id()]] = node_category_t::end;
                } else {
                    node_category[nodes[it->id()]] = node_category_t::regular;
                }
            } else {   // interior angle grater than \pi
                if (below(prev, curr) && below(next, curr)) {
                    node_category[nodes[it->id()]] = node_category_t::split;
                } else if (below(curr, prev) && below(curr, next)) {
                    node_category[nodes[it->id()]] = node_category_t::merge;
                } else {
                    node_category[nodes[it->id()]] = node_category_t::regular;
                }
            }
        }
        // O(nlog(n)) y-coordinate sort (break tiles using x-coordinate)
        std::sort(nodes.begin(), nodes.end(), [&](halfedge_ptr_t n, halfedge_ptr_t m) {
            int i = n->node()->id(), j = m->node()->id();
            return coords(i, 1) > coords(j, 1) || (coords(i, 1) == coords(j, 1) && coords(i, 0) > coords(j, 0));
        });
        // monotone partitioning algorithm (details in section 3.2 of De Berg, M. (2000). Computational geometry:
        // algorithms and applications. Springer Science & Business Media.)
        std::vector<typename std::set<edge_t>::iterator> sweep_line_it(n_nodes);
        sweep_line_it.resize(n_nodes);
        for (halfedge_ptr_t v : nodes) {   // loops in decreasing y-coordinate order
            int prev = v->node()->prev()->id();
            int curr = v->node()->id();
	    int next = v->node()->next()->id();
            // process i-th node
            switch (node_category[v]) {
            case node_category_t::start: {
                auto ref = sweep_line.emplace(curr, coords(curr, 0), coords(curr, 1), coords(next, 0), coords(next, 1));
                sweep_line_it[curr] = ref.first;
                helper[curr] = v;
                break;
            }
            case node_category_t::end: {
                if (node_category[helper[prev]] == node_category_t::merge) { dcel.insert_edge(v, helper[prev]); }
                sweep_line.erase(sweep_line_it[prev]);
                break;
            }
            case node_category_t::split: {
                // search edge in sweep line directly left to curr
                edge_t edge(coords(curr, 0), coords(curr, 1), coords(curr, 0), coords(curr, 1));
                auto it = sweep_line.lower_bound(edge);
                // insert diagonal (update v to point to the inserted diagonal)
                helper[it->id] = dcel.insert_edge(v, helper[it->id]);
                auto ref = sweep_line.emplace(curr, coords(curr, 0), coords(curr, 1), coords(next, 0), coords(next, 1));
                sweep_line_it[curr] = ref.first;
                helper[curr] = v;
                break;
            }
            case node_category_t::merge: {
                if (node_category[helper[prev]] == node_category_t::merge) { dcel.insert_edge(v, helper[prev]); }
                sweep_line.erase(sweep_line_it[prev]);
                // search edge in sweep line directly left to curr
                edge_t edge(coords(curr, 0), coords(curr, 1), coords(curr, 0), coords(curr, 1));
                auto it = sweep_line.lower_bound(edge);
                if (node_category[helper[it->id]] == node_category_t::merge) {
                    v = dcel.insert_edge(v, helper[it->id]);   // update v to point to the inserted diagonal
                }
                helper[it->id] = v;
                break;
            }
            case node_category_t::regular: {
                if (!below(coords.row(curr), coords.row(next))) {
                    // polygon interior is on the right of this halfedge
                    if (node_category[helper[prev]] == node_category_t::merge) { dcel.insert_edge(v, helper[prev]); }
                    sweep_line.erase(sweep_line_it[prev]);
                    auto ref =
                      sweep_line.emplace(curr, coords(curr, 0), coords(curr, 1), coords(next, 0), coords(next, 1));
                    sweep_line_it[curr] = ref.first;
                    helper[curr] = v;
                } else {
                    // search edge in sweep line directly left to curr
                    edge_t edge(coords(curr, 0), coords(curr, 1), coords(curr, 0), coords(curr, 1));
                    auto it = sweep_line.lower_bound(edge);
                    if (node_category[helper[it->id]] == node_category_t::merge) {
                        v = dcel.insert_edge(v, helper[it->id]);
                    }
                    helper[it->id] = v;
                }
                break;
            }
            }
        }
        // recover from the DCEL structure the node numbering of each monotone polygon
        std::vector<bool> visited(dcel.n_halfedges(), false);
        std::vector<std::vector<int>> monotone_partition;
        if (dcel.n_cells() == 1) {   // polygon was already monotone
	  auto& it = monotone_partition.emplace_back();
	  it.resize(n_nodes);
	  std::iota(it.begin(), it.end(), 0);
        } else {
            for (auto cell = dcel.cells_begin(); cell != dcel.cells_end(); ++cell) {
                auto& it = monotone_partition.emplace_back();
                halfedge_t *chain = cell->halfedge(), *end = chain;
                do {
                    it.push_back(chain->node()->id());
                    chain = chain->next();
                } while (end != chain);
            }
        }
        return monotone_partition;
    }
    // triangulate monotone polygon (returns a RowMajor ordered matrix of cells).
    std::vector<int> triangulate_monotone_(const Eigen::Matrix<double, Dynamic, Dynamic>& nodes) {
        // every triangulation of a polygon of n points has n - 2 triangles (lemma 1.2.2 of (1))
        std::vector<int> cells;
	int n_nodes = nodes.rows();
        cells.reserve(3 * (n_nodes - 2));
        auto push_cell = [&](int i, int j, int k) {   // convinient lambda to add a triangle
            cells.push_back(i);
            cells.push_back(j);
            cells.push_back(k);
        };
        if (nodes.rows() == 3) {   // already a triangle, stop
            push_cell(0, 1, 2);
            return cells;
        }
        // O(n) minimum and maximum coordinate search
        int min = nodes(0, 1) < nodes(1, 1) ? 0 : 1, max = nodes(0, 1) < nodes(1, 1) ? 1 : 0;
        double y_min = nodes(min, 1), y_max = nodes(max, 1);
        for (int i = 1; i < n_nodes; ++i) {
            if (nodes(i, 1) < y_min) {
                y_min = nodes(i, 1);
                min = i;
            } else if (nodes(i, 1) > y_max) {
                y_max = nodes(i, 1);
                max = i;
            }
        }
        // build right and left chains (assume counterclockwise sorting)
	std::vector<int> node(n_nodes);
	std::unordered_set<int> r_chain, l_chain;
        {
            std::vector<int> r_chain_, l_chain_;
            for (int j = max; j != min; j = ((j + 1) % n_nodes + n_nodes) % n_nodes) { l_chain_.push_back(j); }
            l_chain_.push_back(min);
            for (int j = ((max - 1) % n_nodes + n_nodes) % n_nodes; j != min;
                 j = ((j - 1) % n_nodes + n_nodes) % n_nodes) {
                r_chain_.push_back(j);
            }
            // O(n) sorted list merge
            std::merge(
              l_chain_.begin(), l_chain_.end(), r_chain_.begin(), r_chain_.end(), node.begin(), [&](int a, int b) {
                  return nodes(a, 1) > nodes(b, 1) || (nodes(a, 1) == nodes(b, 1) && nodes(a, 0) > nodes(b, 0));
              });
            l_chain.insert(l_chain_.begin(), l_chain_.end());
            r_chain.insert(r_chain_.begin(), r_chain_.end());
        }
        auto is_l_chain = [&](int i) { return l_chain.contains(i); };
        auto is_r_chain = [&](int i) { return r_chain.contains(i); };
        auto are_in_opposite_chains = [&](int i, int j) -> bool {
            return (is_l_chain(i) && is_r_chain(j)) || (is_r_chain(i) && is_l_chain(j));
        };
        std::deque<int> reflex_chain;   // queue of reflex nodes
        reflex_chain.push_back(node[0]);
        reflex_chain.push_back(node[1]);
        int node_i, node_j, node_k;
        bool on_left = is_l_chain(node[1]);   // whether the currently pointed node is on the left or right chain
        // start triangulating
        for (std::size_t j = 2, n = node.size(); j < n; ++j) {
            node_i = *(reflex_chain.end() - 1);
            node_j = node[j];
            if (are_in_opposite_chains(node_i, node_j)) {   // triangulate
                node_i = *reflex_chain.begin();
                on_left = !on_left;
                while (reflex_chain.size() > 1) {
                    reflex_chain.pop_front();
                    node_k = reflex_chain.front();
                    // add triangle
		    push_cell(node_i, node_j, node_k);
                    node_i = node_k;
                }
                reflex_chain.push_back(node_j);
            } else {   // check if the triplet (node_i, node_j, node_k) makes a reflex turn or not
                node_k = *(reflex_chain.end() - 2);
                double m_signed =
                  internals::signed_measure_2d_tri(nodes.row(node_j), nodes.row(node_k), nodes.row(node_i));
                if ((on_left && m_signed < 0) || (!on_left && m_signed > 0)) {    // reflex turn
                    reflex_chain.push_back(node_j);
                } else {
                    do {
                        // add triangle
                        push_cell(node_i, node_j, node_k);
                        // triangulate until convex turn is found
                        reflex_chain.pop_back();
                        if (reflex_chain.size() > 1) {
                            node_i = *(reflex_chain.end() - 1);
                            node_k = *(reflex_chain.end() - 2);
                            m_signed =
                              internals::signed_measure_2d_tri(nodes.row(node_j), nodes.row(node_k), nodes.row(node_i));
                        }
                    } while (reflex_chain.size() > 1 && ((on_left && m_signed > 0) || (!on_left && m_signed < 0)));
                    reflex_chain.push_back(node_j);
                }
            }
        }
        return cells;
    }
    // internal face-based storage
    Triangulation<LocalDim, EmbedDim> triangulation_;
};

// a multipolygon is a collection of polygons, backed by a single face-based triangulation
template <int LocalDim, int EmbedDim> class MultiPolygon {
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;
    using polygon_t = Polygon<local_dim, embed_dim>;

    MultiPolygon() : n_polygons_(0) { }
    // rings is a vector of matrix of coordinates, where, each inner matrix defines a closed non self-intersecting loop
    MultiPolygon(const std::vector<Eigen::Matrix<double, Dynamic, Dynamic>>& rings) : n_polygons_(0) {
        // ESRI shapefile format specification: loops in clockwise order define the outer border of a polygon,
        // loops in counterclockwise order defines a hole inside the last found outer polygonal ring
        int n_rings = rings.size();
        if (n_rings == 1) {
            triangulation_ = Polygon<local_dim, embed_dim>(rings[0]).triangulation();
        } else {
            // detect polygons with holes
            using iterator = typename std::vector<Eigen::Matrix<double, Dynamic, Dynamic>>::const_iterator;
            constexpr int n_nodes_per_cell = 3;
            std::vector<std::pair<iterator, iterator>> polygons;
            iterator begin = rings.begin();
            iterator end = begin;
            int n_nodes = 0;
            while (end != rings.end()) {
                n_nodes += end->rows();
                ++end;
                while (end != rings.end() && internals::are_2d_counterclockwise_sorted(*end)) {   // detect holes
                    n_nodes += end->rows();
                    ++end;
                }
                polygons.emplace_back(begin, end);
                begin = end;
            }
            Eigen::Matrix<double, Dynamic, Dynamic> nodes(n_nodes, embed_dim);
            std::vector<int> cells;
            int nodes_off_ = 0;
            for (const Eigen::Matrix<double, Dynamic, Dynamic>& coords : rings) {
                // triangulate polygon
                Triangulation<local_dim, embed_dim> poly_tri = Polygon<local_dim, embed_dim>(coords).triangulation();
                nodes.middleRows(nodes_off_, poly_tri.n_nodes()) = poly_tri.nodes();
                // copy cells
                for (int i = 0, n = poly_tri.cells().rows(); i < n; ++i) {
                    for (int j = 0; j < n_nodes_per_cell; ++j) { cells.push_back(poly_tri.cells()(i, j) + nodes_off_); }
                }
                nodes_off_ += poly_tri.n_nodes();
		n_polygons_++;
		n_nodes_per_polygon_.push_back(poly_tri.n_nodes());
            }
            // set up face-based storage
            triangulation_ = Triangulation<local_dim, embed_dim>(
              nodes,
              Eigen::Map<Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
                cells.data(), cells.size() / n_nodes_per_cell, n_nodes_per_cell),
              Eigen::Matrix<int, Dynamic, 1>::Ones(nodes.rows()));
        }
    }
    // observers
    const Eigen::Matrix<double, Dynamic, Dynamic>& nodes() const { return triangulation_.nodes(); }
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& cells() const { return triangulation_.cells(); }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> edges() const {
        return triangulation_.edges();
    }
    // computes only the boundary edges of the multipoligon (discards triangulation's edges)
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> boundary_edges() const {
        int n_edges = triangulation_.n_boundary_edges();
	Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> m(n_edges, 2);
        int j = 0;
        for (int i = 0, k = triangulation_.n_edges(); i < k; ++i) {
            if (triangulation_.is_edge_on_boundary(i)) { m.row(j++) = triangulation_.edges().row(i); }
        }
        return m;
    }
    const Triangulation<local_dim, embed_dim>& triangulation() const { return triangulation_; }
    double measure() const { return triangulation_.measure(); }
    int n_nodes() const { return triangulation_.n_nodes(); }
    int n_cells() const { return triangulation_.n_cells(); }
    int n_edges() const { return triangulation_.n_edges(); }
    int n_boundary_edges() const { return triangulation_.n_boundary_nodes(); }
    int n_polygons() const { return n_polygons_; }
    // test whether point p is contained in polygon
    template <int Rows, int Cols>
        requires((Rows == embed_dim && Cols == 1) || (Cols == embed_dim && Rows == 1))
    bool contains(const Eigen::Matrix<double, Rows, Cols>& p) const {
        return triangulation_.locate(p) != -1;
    }
    // random sample points in polygon
    Eigen::Matrix<double, Dynamic, Dynamic> sample(int n_samples, int seed = random_seed) const {
        return triangulation_.sample(n_samples, seed);
    }
   private:
    // internal face-based storage
    Triangulation<LocalDim, EmbedDim> triangulation_;
    std::vector<int> n_nodes_per_polygon_;
    int n_polygons_;
};

}   // namespace fdapde

#endif // __FDAPDE_POLYGON_H__
