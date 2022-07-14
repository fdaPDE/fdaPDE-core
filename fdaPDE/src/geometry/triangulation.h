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

#ifndef __FDAPDE_TRIANGULATION_H__
#define __FDAPDE_TRIANGULATION_H__

#include "header_check.h"

namespace fdapde {

// public iterator types
template <typename Triangulation> struct CellIterator : public Triangulation::cell_iterator {
    CellIterator(int index, const Triangulation* mesh) : Triangulation::cell_iterator(index, mesh) { }
    CellIterator(int index, const Triangulation* mesh, int marker) :
        Triangulation::cell_iterator(index, mesh, marker) { }
};
template <typename Triangulation> struct BoundaryIterator : public Triangulation::boundary_iterator {
    BoundaryIterator(int index, const Triangulation* mesh) : Triangulation::boundary_iterator(index, mesh) { }
    BoundaryIterator(int index, const Triangulation* mesh, int marker) :
        Triangulation::boundary_iterator(index, mesh, marker) { }
};
  
template <int LocalDim, int EmbedDim> class Triangulation;
template <int LocalDim, int EmbedDim, typename Derived> class TriangulationBase {
    using dbl_matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using int_matrix_t = Eigen::Matrix<int, Dynamic, Dynamic>;
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;
    static constexpr int n_nodes_per_cell = local_dim + 1;
    static constexpr bool is_manifold = !(local_dim == embed_dim);
    using CellType = std::conditional_t<
      local_dim == 1, Segment<Derived>, std::conditional_t<local_dim == 2, Triangle<Derived>, Tetrahedron<Derived>>>;
    using TriangulationType = Derived;
    class NodeType {   // triangulation node abstraction
        int id_;
        const Derived* mesh_;
       public:
        NodeType() = default;
        NodeType(int id, const Derived* mesh) : id_(id), mesh_(mesh) { }
        int id() const { return id_; }
        Eigen::Matrix<double, embed_dim, 1> coords() const { return mesh_->node(id_); }
        std::vector<int> patch() const { return mesh_->node_patch(id_); }         // cells having this node as vertex
        std::vector<int> one_ring() const { return mesh_->node_one_ring(id_); }   // directly connected nodes
    };

    TriangulationBase() = default;
    TriangulationBase(const dbl_matrix_t& nodes, const int_matrix_t& cells, const int_matrix_t& boundary, int flags) :
        nodes_(nodes), cells_(cells), boundary_markers_(boundary), flags_(flags) {
        fdapde_assert(
          nodes.rows() > 0 && nodes.cols() == embed_dim && cells.rows() > 0 && cells.cols() == n_nodes_per_cell &&
          boundary.rows() == nodes.rows() && boundary.cols() == 1);
        fdapde_assert(cells.minCoeff() >= 0);
	int min = cells.minCoeff();
        if (min != 0) { cells_ = cells_.array() - min; }   // scale cell numbering to start from zero
        // store number of nodes and number of cells
        n_nodes_ = nodes_.rows();
        n_cells_ = cells_.rows();
        // compute mesh bounding box
        bbox_.row(0) = nodes_.colwise().minCoeff();
        bbox_.row(1) = nodes_.colwise().maxCoeff();
	// set-up markers
	cells_markers_.resize(n_cells_, Unmarked);
	nodes_markers_.resize(n_nodes_, Unmarked);
    }
    TriangulationBase(const dbl_matrix_t& nodes, const int_matrix_t& cells, const int_matrix_t& boundary) :
        TriangulationBase(nodes, cells, boundary, /* flags = */ 0) { }
    TriangulationBase(
      const std::string& nodes, const std::string& cells, const std::string& boundary, bool header, bool index_col,
      int flags) :
        TriangulationBase(
          read_table<double>(nodes, header, index_col).as_matrix(),
          read_table<int>(cells, header, index_col).as_matrix(),
          read_table<int>(boundary, header, index_col).as_matrix(), flags) { }
    TriangulationBase(
      const std::string& nodes, const std::string& cells, const std::string& boundary, bool header, bool index_col) :
        TriangulationBase(nodes, cells, boundary, header, index_col, /* flags = */ 0) { }

    // getters
    Eigen::Matrix<double, embed_dim, 1> node(int id) const { return nodes_.row(id); }
    bool is_node_on_boundary(int id) const { return boundary_markers_[id]; }
    const Eigen::Matrix<double, Dynamic, Dynamic>& nodes() const { return nodes_; }
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& cells() const { return cells_; }
    const BinaryVector<Dynamic>& boundary_nodes() const { return boundary_markers_; }
    int n_cells() const { return n_cells_; }
    int n_nodes() const { return n_nodes_; }
    int n_boundary_nodes() const { return boundary_markers_.count(); }
    Eigen::Matrix<double, 2, embed_dim> bbox() const { return bbox_; }
    double measure() const {
        return std::accumulate(
          cells_begin(), cells_end(), 0.0, [](double v, const auto& e) { return v + e.measure(); });
    }
    // iterators over cells (possibly filtered by marker)
    class cell_iterator : public internals::filtering_iterator<cell_iterator, const CellType*> {
        using Base = internals::filtering_iterator<cell_iterator, const CellType*>;
        using Base::index_;
        friend Base;
        const Derived* mesh_;
        int marker_;
        cell_iterator& operator()(int i) {
            Base::val_ = &(mesh_->cell(i));
            return *this;
        }
       public:
        using TriangulationType = Derived;
        cell_iterator() = default;
        cell_iterator(int index, const Derived* mesh, const BinaryVector<Dynamic>& filter, int marker) :
            Base(index, 0, mesh->n_cells_, filter), mesh_(mesh), marker_(marker) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        cell_iterator(int index, const Derived* mesh, int marker) :
            cell_iterator(
              index, mesh,
              marker == TriangulationAll ? BinaryVector<Dynamic>::Ones(mesh->n_cells()) :   // apply no filter
                make_binary_vector(mesh->cells_markers().begin(), mesh->cells_markers().end(), marker),
              marker) { }
        int marker() const { return marker_; }
    };
    CellIterator<Derived> cells_begin(int marker = TriangulationAll) const {
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers_.size() != 0));
        return CellIterator<Derived>(0, static_cast<const Derived*>(this), marker);
    }
    CellIterator<Derived> cells_end(int marker = TriangulationAll) const {
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers_.size() != 0));
        return CellIterator<Derived>(n_cells_, static_cast<const Derived*>(this), marker);
    }
    // set cells markers
    template <typename Lambda>
    void mark_cells(int marker, Lambda&& lambda)
        requires(requires(Lambda lambda, CellType c) {
            { lambda(c) } -> std::same_as<bool>;
        }) {
        fdapde_assert(marker >= 0);
        cells_markers_.resize(n_cells_, Unmarked);
        for (cell_iterator it = cells_begin(); it != cells_end(); ++it) {
            cells_markers_[it->id()] = lambda(*it) ? marker : Unmarked;
        }
    }
    template <int Rows, typename XprType> void mark_cells(const BinMtxBase<Rows, 1, XprType>& mask) {
        fdapde_assert(mask.rows() == n_cells_);
        cells_markers_.resize(n_cells_, Unmarked);
        for (cell_iterator it = cells_begin(); it != cells_end(); ++it) {
            cells_markers_[it->id()] = mask[it->id()] ? 1 : 0;
        }
    }
    template <typename Iterator> void mark_cells(Iterator first, Iterator last) {
        fdapde_static_assert(
          std::is_convertible_v<typename Iterator::value_type FDAPDE_COMMA int>, INVALID_ITERATOR_RANGE);
        int n_markers = std::distance(first, last);
        bool all_markers_positive = std::all_of(first, last, [](auto marker) { return marker >= 0; });
        fdapde_assert(n_markers == n_cells_ && all_markers_positive);
        cells_markers_.resize(n_cells_, Unmarked);
        for (int i = 0; i < n_cells_; ++i) { cells_markers_[i] = *(first + i); }
    }
    void mark_cells(int marker) {   // marks all cells with m
        fdapde_assert(marker >= 0);
        cells_markers_.resize(n_cells_);
	std::for_each(cells_markers_.begin(), cells_markers_.end(), [marker](int& marker_) { marker_ = marker; });
    }
    void clear_cell_markers() {
        std::for_each(cells_markers_.begin(), cells_markers_.end(), [](int& marker) { marker = Unmarked; });
    }
    const std::vector<int>& cells_markers() const { return cells_markers_; }
    const std::vector<int>& nodes_markers() const { return nodes_markers_; }
    // iterator over boundary nodes
    class boundary_node_iterator : public internals::filtering_iterator<boundary_node_iterator, NodeType> {
        using Base = internals::filtering_iterator<boundary_node_iterator, NodeType>;
        using Base::index_;
        friend Base;
        const Derived* mesh_;
        int marker_;
        boundary_node_iterator& operator()(int i) {
            Base::val_ = NodeType(i, mesh_);
            return *this;
        }
       public:
        using TriangulationType = Derived;
        boundary_node_iterator() = default;
        boundary_node_iterator(int index, const Derived* mesh, int marker) :
            Base(
              index, 0, mesh->n_nodes_,
              marker == BoundaryAll ? mesh->boundary_markers_ :   // apply no custom filter
                make_binary_vector(mesh->nodes_markers().begin(), mesh->nodes_markers().end(), marker) &
                  mesh->boundary_markers_),
            mesh_(mesh) {
            for (; index_ < Base::end_ && !Base::filter_[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
    };
    boundary_node_iterator boundary_nodes_begin(int marker = BoundaryAll) const {
        return boundary_node_iterator(0, static_cast<const Derived*>(this), marker);
    }
    boundary_node_iterator boundary_nodes_end(int marker = BoundaryAll) const {
        return boundary_node_iterator(n_nodes_, static_cast<const Derived*>(this), marker);
    }
    // random sample points in triangulation
    Eigen::Matrix<double, Dynamic, Dynamic> sample(int n_samples, int seed = random_seed) const {
        // set up random number generation
        int seed_ = (seed == random_seed) ? std::random_device()() : seed;
        std::mt19937 rng(seed_);
        // probability of random sampling a cell equals (measure of cell)/(measure of polygon)
        std::vector<double> weights(n_cells_);
        for (cell_iterator it = cells_begin(); it != cells_end(); ++it) { weights[it->id()] = it->measure(); }
        std::discrete_distribution<int> rand_cell(weights.begin(), weights.end());
        std::uniform_real_distribution<double> rand_real(0, 1);
        Eigen::Matrix<double, Dynamic, Dynamic> coords(n_samples, embed_dim);
        for (int i = 0; i < n_samples; ++i) {
            // generate random cell
            int cell_id = rand_cell(rng);
            auto e = static_cast<const Derived&>(*this).cell(cell_id);
            // generate random point in cell
            double t = rand_real(rng);
            coords.row(i) = t * e.node(0) + (1 - t) * e.node(1);
            for (int j = 1; j < local_dim; ++j) {
                t = rand_real(rng);
                coords.row(i) = (1 - t) * e.node(1 + j).transpose() + t * coords.row(i);
            }
        }
        return coords;
    }
    // all nodes directly connected to node having id node_id, O(log(n))
    std::vector<int> node_one_ring(int node_id) const {
        std::vector<int> patch_ = static_cast<const Derived&>(*this).node_patch(node_id);   // O(log(n)) step
        std::unordered_set<int> one_ring;
        for (int cell_id : patch_) {
            for (int i = 0; i < n_nodes_per_cell; ++i) { one_ring.insert(cells_(cell_id, i)); }
        }
        return {one_ring.begin(), one_ring.end()};
    }
   protected:
    Eigen::Matrix<double, Dynamic, Dynamic> nodes_ {};                 // physical coordinates of mesh's vertices
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> cells_ {};   // nodes composing each cell
    BinaryVector<Dynamic> boundary_markers_ {};                        // j-th element is 1 \iff node j is on boundary
    Eigen::Matrix<double, 2, embed_dim> bbox_ {};   // mesh bounding box (column i maps to the i-th dimension)
    int n_nodes_ = 0, n_cells_ = 0;
    int flags_ = 0;
    std::vector<int> cells_markers_;   // marker associated to i-th cell
    std::vector<int> nodes_markers_;   // marker associated to i-th node
};

// face-based storage
template <int N> class Triangulation<2, N> : public TriangulationBase<2, N, Triangulation<2, N>> {
    fdapde_static_assert(N == 2 || N == 3, THIS_CLASS_IS_FOR_2D_OR_3D_TRIANGULATIONS_ONLY);
   public:
    using Base = TriangulationBase<2, N, Triangulation<2, N>>;
    static constexpr int n_nodes_per_edge = 2;
    static constexpr int n_edges_per_cell = 3;
    static constexpr int n_faces_per_edge = 2;
    static constexpr int n_neighbors_per_cell = 3;
    using EdgeType = typename Base::CellType::EdgeType;
    using LocationPolicy = TreeSearch<Triangulation<2, N>>;
    using Base::cells_;      // N \times 3 matrix of node identifiers for each triangle
    using Base::embed_dim;   // dimensionality of the ambient space
    using Base::local_dim;   // dimensionality of the tangent space
    using Base::n_cells_;    // N: number of triangles
    using Base::n_nodes_per_cell;
    static constexpr auto edge_pattern =
      Matrix<int, binomial_coefficient(n_nodes_per_cell, n_nodes_per_edge), n_nodes_per_edge>(
        combinations(n_nodes_per_edge, n_nodes_per_cell));

    Triangulation() = default;
    Triangulation(
      const Eigen::Matrix<double, Dynamic, Dynamic>& nodes, const Eigen::Matrix<int, Dynamic, Dynamic>& cells,
      const Eigen::Matrix<int, Dynamic, Dynamic>& boundary, int flags = 0) :
        Base(nodes, cells, boundary, flags) {
        // -1 in neighbors_'s column i implies no neighbor adjacent to the edge opposite to vertex i
        neighbors_ = Eigen::Matrix<int, Dynamic, Dynamic>::Constant(n_cells_, n_neighbors_per_cell, -1);

        if (Base::flags_ & cache_cells) {   // populate cache if cell caching is active
            cell_cache_.reserve(n_cells_);
            for (int i = 0; i < n_cells_; ++i) { cell_cache_.emplace_back(i, this); }
        }
        using edge_t = std::array<int, n_nodes_per_edge>;
        using hash_t = internals::std_array_hash<int, n_nodes_per_edge>;
        struct edge_info {
            int edge_id, face_id;   // for each edge, its ID and the ID of one of the cells insisting on it
        };
        std::unordered_map<edge_t, edge_info, hash_t> edges_map;
	std::vector<bool> boundary_edges;
        edge_t edge;
        cell_to_edges_.resize(n_cells_, n_edges_per_cell);
        // search vertex of face f opposite to edge e (the j-th vertex of f which is not a node of e)
        auto node_opposite_to_edge = [this](int e, int f) -> int {
            int j = 0;
            for (; j < Base::n_nodes_per_cell; ++j) {
                bool found = false;
                for (int k = 0; k < n_nodes_per_edge; ++k) {
                    if (edges_[e * n_nodes_per_edge + k] == cells_(f, j)) { found = true; }
                }
                if (!found) break;
            }
            return j;
        };
        int edge_id = 0;
        for (int i = 0; i < n_cells_; ++i) {
            for (int j = 0; j < edge_pattern.rows(); ++j) {
                // construct edge
                for (int k = 0; k < n_nodes_per_edge; ++k) { edge[k] = cells_(i, edge_pattern(j, k)); }
                std::sort(edge.begin(), edge.end());   // normalize wrt node ordering
                auto it = edges_map.find(edge);
                if (it == edges_map.end()) {   // never processed edge
                    edges_.insert(edges_.end(), edge.begin(), edge.end());
                    edge_to_cells_.insert(edge_to_cells_.end(), {i, -1});
                    boundary_edges.push_back(true);
                    edges_map.emplace(edge, edge_info {edge_id, i});
                    cell_to_edges_(i, j) = edge_id;
                    edge_id++;
                } else {
                    const auto& [h, k] = it->second;
                    // elements k and i are neighgbors (they share an edge)
                    this->neighbors_(k, node_opposite_to_edge(h, k)) = i;
                    this->neighbors_(i, node_opposite_to_edge(h, i)) = k;
                    cell_to_edges_(i, j) = h;
                    boundary_edges[h] = false;   // edge_id-th edge cannot be on boundary
                    edge_to_cells_[2 * h + 1] = i;
                    edges_map.erase(it);
                }
            }
        }
        n_edges_ = edges_.size() / n_nodes_per_edge;
        boundary_edges_ = BinaryVector<Dynamic>(boundary_edges.begin(), boundary_edges.end(), n_edges_);
        return;
    }
    Triangulation(
      const std::string& nodes, const std::string& cells, const std::string& boundary, bool header, bool index_col,
      int flags = 0) :
        Triangulation(
          read_table<double>(nodes, header, index_col).as_matrix(),
          read_table<int>(cells, header, index_col).as_matrix(),
          read_table<int>(boundary, header, index_col).as_matrix(), flags) { }
    // static constructors
    static Triangulation<2, N>
    Rectangle(double a_x, double b_x, double a_y, double b_y, int nx, int ny, int flags = 0) {
        fdapde_static_assert(N == 2, THIS_METHOD_IS_ONLY_FOR_TWO_DIMENSIONAL_TRIANGULATIONS);
        // allocate memory
        Eigen::Matrix<double, Dynamic, Dynamic> nodes(nx * ny, 2);
        Eigen::Matrix<int, Dynamic, Dynamic> cells((nx - 1) * (ny - 1) * 2, 3);   // each subrectangle splitted in 2
        Eigen::Matrix<int, Dynamic, Dynamic> boundary(nx * ny, 1);
        // compute steps along axis
        double h_x = (b_x - a_x) / (nx - 1);
        double h_y = (b_y - a_y) / (ny - 1);
        int cell_id = 0;
        std::array<int, 4> v;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int node_id = j * nx + i;
                nodes.row(node_id) << (a_x + i * h_x), (a_y + j * h_y);   // node coordinates
                if (i < nx - 1 && j < ny - 1) {
                    // compute vector of vertices of subrectangle
                    int p = j * nx + i;
                    v = {p, p + 1, p + nx, p + nx + 1};
                    // compute vertices of each triangle in the subrectangle
                    cells.row(cell_id + 0) << v[0], v[1], v[2];
                    cells.row(cell_id + 1) << v[1], v[2], v[3];
                    cell_id = cell_id + 2;
                }
                boundary(node_id, 0) = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) ? 1 : 0;
            }
        }
        return Triangulation<2, N>(nodes, cells, boundary, flags);
    }
    static Triangulation<2, N> Square(double a, double b, int n_nodes, int flags = 0) {
        return Triangulation<2, N>::Rectangle(a, b, a, b, n_nodes, n_nodes, flags);
    }
    static Triangulation<2, N> UnitSquare(int n_nodes, int flags = 0) { return Square(0.0, 1.0, n_nodes, flags); }
    // icoshpere surface generation
    static Triangulation<2, N> Sphere(double radius, int n_refinments, int flags = 0) {
        fdapde_static_assert(N == 3, THIS_METHOD_IS_FOR_THREE_DIMENSIONAL_SURFACE_TRIANGULATIONS_ONLY);
        fdapde_assert(radius > 0 && n_refinments > 0);
        // unit icosahedron construction
        constexpr double a = 1.0;
        constexpr double b = 1.0 / std::numbers::phi;   // inverse golden ratio
        std::vector<double> ico_nodes = {
	  -a, 0, b,  a, 0,  b, -a,  0, -b,  a,  0, -b,
	   0, b, a,  0, b, -a,  0, -b,  a,  0, -b, -a,
	   b, a, 0, -b, a,  0,  b, -a,  0, -b, -a,  0};
	std::vector<int> ico_cells = {
	   0, 4,  6, 0,  6, 11, 0, 2, 11, 0,  9, 2, 0,  9,  4,  3,  8, 5, 3,  5, 7,
	   3, 7, 10, 3, 10,  1, 3, 1,  8, 1, 10, 6, 6, 11, 10, 10, 11, 7, 7, 11, 2,
	   7, 2,  5, 2,  5,  9, 9, 5,  8, 9,  4, 8, 4,  8,  1,  4,  1, 6};
        // normalize to unit sphere
        constexpr double norm = std::sqrt(a * a + b * b);
        std::for_each(ico_nodes.begin(), ico_nodes.end(), [](double& v) { v /= norm; });

	// refinment
        using vec_t = Eigen::Matrix<double, embed_dim, 1>;
        using edge_t = std::array<int, n_nodes_per_edge>;
        using hash_t = internals::std_array_hash<int, n_nodes_per_edge>;
        std::unordered_map<edge_t, int, hash_t> edges_map;   // for each edge, the ID of its midpoint
        edge_t edge;
        // mesh construction
        std::vector<double> nodes = ico_nodes;
        int node_id = nodes.size() / n_nodes_per_cell;
        std::vector<int> cells = ico_cells, refined_cells;
        for (int i = 0; i < n_refinments; ++i) {
            refined_cells.clear();
            edges_map.clear();
	    // split each triangle
            for (int cell_id = 0; cell_id < cells.size() / n_nodes_per_cell; ++cell_id) {
                // Loop subdivision algorithm with spherical projection
                std::vector<int> n(6);
                for (int j = 0; j < n_nodes_per_cell; ++j) { n[j] = cells[cell_id * n_nodes_per_cell + j]; }
                for (int j = 0; j < edge_pattern.rows(); ++j) {
                    // construct edge
                    for (int k = 0; k < n_nodes_per_edge; ++k) {
                        edge[k] = cells[cell_id * n_nodes_per_cell + edge_pattern(j, k)];
                    }
                    std::sort(edge.begin(), edge.end());   // normalize wrt node ordering
                    auto it = edges_map.find(edge);
                    if (it == edges_map.end()) {   // never processed edge
                        edges_map.emplace(edge, node_id);
                        n[n_nodes_per_cell + j] = node_id;
                        node_id++;
                        // compute slerp interpolation
                        double dot_prod = 0;
                        for (int i = 0; i < embed_dim; ++i) {
                            dot_prod += nodes[edge[0] * embed_dim + i] * nodes[edge[1] * embed_dim + i];
                        }
                        double rho = std::acos(dot_prod);
                        // midpoint along circular arc
                        double a = std::sin(0.5 * rho), b = std::sin(rho);
                        for (int i = 0; i < embed_dim; ++i) {
                            nodes.push_back(a / b * (nodes[edge[0] * embed_dim + i] + nodes[edge[1] * embed_dim + i]));
                        }
                    } else {
                        n[n_nodes_per_cell + j] = edges_map.at(edge);
                    }
                }
                // compute cell numbering (exploit edge_ordering structure)
                std::vector<int> cvec = {
		  n[0], n[3], n[4],
		  n[1], n[3], n[5],
		  n[5], n[2], n[4],
		  n[4], n[3], n[5]
		};
                refined_cells.insert(refined_cells.end(), cvec.begin(), cvec.end());
            }
            cells = refined_cells;
        }
        Eigen::Map<Eigen::Matrix<double, Dynamic, Dynamic, Eigen::RowMajor>> nodes_mtx(
          nodes.data(), nodes.size() / embed_dim, embed_dim);
	// scale to requested radius
	if(radius != 1.0) { nodes_mtx *= radius; }
        Eigen::Map<Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> cells_mtx(
          refined_cells.data(), refined_cells.size() / n_nodes_per_cell, n_nodes_per_cell);
        Eigen::Matrix<int, Dynamic, Dynamic> boundary = Eigen::Matrix<int, Dynamic, Dynamic>::Zero(nodes_mtx.rows(), 1);
        return Triangulation<2, N>(nodes_mtx, cells_mtx, boundary, flags);
    }
    static Triangulation<2, N> UnitSphere(int n_refinments, int flags = 0) { return Sphere(1.0, n_refinments, flags); }
    // getters
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& neighbors() const { return neighbors_; }
    const typename Base::CellType& cell(int id) const {
        if (Base::flags_ & cache_cells) {   // cell caching enabled
            return cell_cache_[id];
        } else {
            cell_ = typename Base::CellType(id, this);
            return cell_;
        }
    }
    bool is_edge_on_boundary(int id) const { return boundary_edges_[id]; }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> edges() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          edges_.data(), n_edges_, n_nodes_per_edge);
    }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> edge_to_cells() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          edge_to_cells_.data(), n_edges_, 2);
    }
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& cell_to_edges() const { return cell_to_edges_; }
    const BinaryVector<Dynamic>& boundary_edges() const { return boundary_edges_; }
    int n_edges() const { return n_edges_; }
    int n_boundary_edges() const { return boundary_edges_.count(); }
    // iterators over edges
    class edge_iterator : public internals::filtering_iterator<edge_iterator, EdgeType> {
       protected:
        using Base = internals::filtering_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const Triangulation* mesh_;
        int marker_;
        edge_iterator& operator()(int i) {
            Base::val_ = EdgeType(i, mesh_);
            return *this;
        }
       public:
        using TriangulationType = Triangulation<2, N>;
        edge_iterator(int index, const Triangulation* mesh, const BinaryVector<Dynamic>& filter, int marker) :
            Base(index, 0, mesh->n_edges_, filter), mesh_(mesh), marker_(marker) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        edge_iterator(int index, const Triangulation* mesh) :   // apply no filter
            edge_iterator(index, mesh, BinaryVector<Dynamic>::Ones(mesh->n_edges_), Unmarked) { }
        edge_iterator(int index, const Triangulation* mesh, int marker) :   // fast construction for end iterators
            Base(index, 0, mesh->n_edges_), marker_(marker) { }
        int marker() const { return marker_; }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(n_edges_, this, Unmarked); }
    // iterator over boundary edges
    struct boundary_edge_iterator : public edge_iterator {
        boundary_edge_iterator(int index, const Triangulation* mesh) :
            edge_iterator(index, mesh, mesh->boundary_edges_, BoundaryAll) { }
        boundary_edge_iterator(int index, const Triangulation* mesh, int marker) :   // filter boundary edges by marker
            edge_iterator(
              index, mesh,
              marker == BoundaryAll ?
                mesh->boundary_edges_ :
                mesh->boundary_edges_ &
                  make_binary_vector(mesh->edges_markers_.begin(), mesh->edges_markers_.end(), marker),
              marker) { }
    };
    boundary_edge_iterator boundary_edges_begin() const { return boundary_edge_iterator(0, this); }
    boundary_edge_iterator boundary_edges_end() const { return boundary_edge_iterator(n_edges_, this); }
    using boundary_iterator = boundary_edge_iterator;   // public view of 2D boundary
    BoundaryIterator<Triangulation<2, N>> boundary_begin(int marker = BoundaryAll) const {
        return BoundaryIterator<Triangulation<2, N>>(0, this, marker);
    }
    BoundaryIterator<Triangulation<2, N>> boundary_end(int marker = BoundaryAll) const {
        return BoundaryIterator<Triangulation<2, N>>(n_edges_, this, marker);
    }
    std::pair<BoundaryIterator<Triangulation<2, N>>, BoundaryIterator<Triangulation<2, N>>>
    boundary(int marker = BoundaryAll) const {
        return std::make_pair(boundary_begin(marker), boundary_end(marker));
    }
    const std::vector<int>& edges_markers() const { return edges_markers_; }
    // set boundary markers
    template <typename Lambda>
    void mark_boundary(int marker, Lambda&& lambda)
        requires(requires(Lambda lambda, EdgeType e) {
            { lambda(e) } -> std::same_as<bool>;
        }) {
        fdapde_assert(marker >= 0);
        edges_markers_.resize(n_edges_);
        for (boundary_edge_iterator it = boundary_edges_begin(); it != boundary_edges_end(); ++it) {
            if (lambda(*it)) {
                edges_markers_[it->id()] = marker;
            }
        }
    }
    template <int Rows, typename XprType> void mark_boundary(const BinMtxBase<Rows, 1, XprType>& mask) {
        fdapde_assert(mask.rows() == n_edges_);
        edges_markers_.resize(n_edges_, 0);
        for (boundary_edge_iterator it = boundary_edges_begin(); it != boundary_edges_end(); ++it) {
            if(mask[it->id()]){
                edges_markers_[it->id()] = 1;
            }  
        }
    }
    template <typename Iterator> void mark_boundary(Iterator first, Iterator last) {
        fdapde_static_assert(
          std::is_convertible_v<typename Iterator::value_type FDAPDE_COMMA int>, INVALID_ITERATOR_RANGE);
        int n_markers = std::distance(first, last);
	bool all_markers_positive = std::all_of(first, last, [](auto marker) { return marker >= 0; });
        fdapde_assert(n_markers == n_edges() && all_markers_positive);
        edges_markers_.resize(n_edges_, Unmarked);
        for (int i = 0; i < n_edges_; ++i) { edges_markers_[i] = *(first + i); }
        return;
    }
    // marks all boundary edges
    void mark_boundary(int marker) {
        fdapde_assert(marker >= 0);
        edges_markers_.resize(n_edges_, Unmarked);
        std::for_each(edges_markers_.begin(), edges_markers_.end(), [marker](int& marker_) { marker_ = marker; });
    }
    void clear_boundary_markers() {
        std::for_each(edges_markers_.begin(), edges_markers_.end(), [](int& marker) { marker = Unmarked; });
    }
    // point location
    template <int Rows, int Cols>
    std::conditional_t<Rows == Dynamic || Cols == Dynamic, Eigen::Matrix<int, Dynamic, 1>, int>
    locate(const Eigen::Matrix<double, Rows, Cols>& p) const {
        fdapde_static_assert(
          (Cols == 1 && Rows == embed_dim) || (Cols == Dynamic && Rows == Dynamic),
          YOU_PASSED_A_MATRIX_OF_POINTS_TO_LOCATE_OF_WRONG_DIMENSIONS);
        if (!location_policy_.has_value()) { location_policy_ = LocationPolicy(this); }
        return location_policy_->locate(p);
    }
    template <typename Derived> Eigen::Matrix<int, Dynamic, 1> locate(const Eigen::Map<Derived>& p) const {
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->locate(p);
    }
    // the set of cells which have node id as vertex
    std::vector<int> node_patch(int id) const {
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->all_locate(Base::node(id));
    }
   protected:
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> neighbors_ {};   // adjacent cells ids (-1: no adjacent cell)
    std::vector<int> edges_ {};           // nodes (as row indexes in nodes_ matrix) composing each edge
    std::vector<int> edge_to_cells_ {};   // for each edge, the ids of adjacent cells
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> cell_to_edges_ {};   // ids of edges composing each cell
    BinaryVector<Dynamic> boundary_edges_ {};   // j-th element is 1 \iff edge j is on boundary
    std::vector<int> edges_markers_ {};
    int n_edges_ = 0;
    mutable std::optional<LocationPolicy> location_policy_ {};
    // cell caching
    std::vector<typename Base::CellType> cell_cache_;
    mutable typename Base::CellType cell_;   // used in case cell caching is off
};

// face-based storage
template <> class Triangulation<3, 3> : public TriangulationBase<3, 3, Triangulation<3, 3>> {
   public:
    using Base = TriangulationBase<3, 3, Triangulation<3, 3>>;
    static constexpr int n_nodes_per_face = 3;
    static constexpr int n_nodes_per_edge = 2;
    static constexpr int n_edges_per_face = 3;
    static constexpr int n_faces_per_cell = 4;
    static constexpr int n_edges_per_cell = 6;
    static constexpr int n_neighbors_per_cell = 4;
    using FaceType = typename Base::CellType::FaceType;
    using EdgeType = typename Base::CellType::EdgeType;
    using LocationPolicy = TreeSearch<Triangulation<3, 3>>;
    using Base::embed_dim;
    using Base::local_dim;
    using Base::n_nodes_per_cell;
    static constexpr auto face_pattern =
      Matrix<int, binomial_coefficient(n_nodes_per_cell, n_nodes_per_face), n_nodes_per_face>(
        combinations(n_nodes_per_face, n_nodes_per_cell));
    static constexpr auto edge_pattern =
      Matrix<int, binomial_coefficient(n_nodes_per_face, n_nodes_per_edge), n_nodes_per_edge>(
        combinations(n_nodes_per_edge, n_nodes_per_face));

    Triangulation() = default;
    Triangulation(
      const Eigen::Matrix<double, Dynamic, Dynamic>& nodes, const Eigen::Matrix<int, Dynamic, Dynamic>& cells,
      const Eigen::Matrix<int, Dynamic, Dynamic>& boundary, int flags = 0) :
        Base(nodes, cells, boundary, flags) {
        // -1 in neighbors_'s column i implies no neighbor adjacent to the edge opposite to vertex i
        neighbors_ = Eigen::Matrix<int, Dynamic, Dynamic>::Constant(n_cells_, n_neighbors_per_cell, -1);

        using face_t = std::array<int, n_nodes_per_face>;
        using edge_t = std::array<int, n_nodes_per_edge>;
        struct face_info {
            int face_id, cell_id;   // for each face, its ID and the ID of one of the faces insisting on it
        };
	using edge_info = int;
        std::unordered_map<edge_t, edge_info, internals::std_array_hash<int, n_nodes_per_edge>> edges_map;
        std::unordered_map<face_t, face_info, internals::std_array_hash<int, n_nodes_per_face>> faces_map;
        std::vector<bool> boundary_faces, boundary_edges;
        face_t face, face_;
        edge_t edge, edge_;
        cell_to_faces_.resize(n_cells_, n_faces_per_cell);
        // search vertex of face f opposite to edge e (the j-th vertex of f which is not a node of e)
        auto node_opposite_to_face = [this](int e, int f) -> int {
            int j = 0;
            for (; j < Base::n_nodes_per_cell; ++j) {
                bool found = false;
                for (int k = 0; k < n_nodes_per_face; ++k) {
                    if (faces_[e * n_nodes_per_face + k] == cells_(f, j)) { found = true; }
                }
                if (!found) break;
            }
            return j;
        };
        int face_id = 0, edge_id = 0;
        for (int i = 0; i < n_cells_; ++i) {
            for (int j = 0; j < face_pattern.rows(); ++j) {
                // construct face. faces are constructed with ordering: (0 1 2), (0 1 3), (0 2 3), (1 2 3)
                for (int k = 0; k < n_nodes_per_face; ++k) { face[k] = cells_(i, face_pattern(j, k)); }
		face_ = face;
                std::sort(face.begin(), face.end());   // normalize wrt node ordering
                auto it = faces_map.find(face);
                if (it == faces_map.end()) {   // never processed face
                    faces_.insert(faces_.end(), face_.begin(), face_.end());
                    face_to_cells_.insert(face_to_cells_.end(), {i, -1});
                    boundary_faces.push_back(true);
                    faces_map.emplace(face, face_info {face_id, i});
                    cell_to_faces_(i, j) = face_id;
                    face_id++;
                    // compute for each face the ids of its edges
                    for (int k = 0; k < n_edges_per_face; ++k) {
                        // construct edge. edges are constructed with order: (0 1), (0 2), (1, 2)
                        for (int h = 0; h < n_nodes_per_edge; ++h) { edge[h] = face_[edge_pattern(k, h)]; }
			edge_ = edge;
                        std::sort(edge.begin(), edge.end());
                        auto it = edges_map.find(edge);
                        if (it == edges_map.end()) {
                            edges_.insert(edges_.end(), edge_.begin(), edge_.end());
                            face_to_edges_.push_back(edge_id);
                            edge_to_cells_[edge_id].insert(i);   // store (edge, cell) binding
                            edges_map.emplace(edge, edge_id);
                            boundary_edges.push_back(boundary_markers_[edge[0]] && boundary_markers_[edge[1]]);
                            edge_id++;
                        } else {
                            face_to_edges_.push_back(edges_map.at(edge));
			    edge_to_cells_[edges_map.at(edge)].insert(i);   // store (edge, cell) binding
                        }
                    }
                } else {
                    const auto& [h, k] = it->second;
                    // elements k and i are neighgbors (they share face with id h)
                    neighbors_(k, node_opposite_to_face(h, k)) = i;
                    neighbors_(i, node_opposite_to_face(h, i)) = k;
		    // store (edge, cell) binding for each edge of this face
                    for (int edge = 0; edge < n_edges_per_face; edge++) {
                        edge_to_cells_.at(face_to_edges_[h * n_edges_per_face + edge]).insert(i);
                    }
                    cell_to_faces_(i, j) = h;
		    face_to_cells_[2 * h + 1] = i;
		    boundary_faces[h] = false;
                    faces_map.erase(it);
                }
            }
        }
        n_faces_ = faces_.size() / n_nodes_per_face;
        n_edges_ = edges_.size() / n_nodes_per_edge;
        boundary_faces_ = BinaryVector<Dynamic>(boundary_faces.begin(), boundary_faces.end(), n_faces_);
        boundary_edges_ = BinaryVector<Dynamic>(boundary_edges.begin(), boundary_edges.end(), n_edges_);
        if (Base::flags_ & cache_cells) {   // populate cache if cell caching is active
            cell_cache_.reserve(n_cells_);
            for (int i = 0; i < n_cells_; ++i) { cell_cache_.emplace_back(i, this); }
        }
        return;
    }
    Triangulation(
      const std::string& nodes, const std::string& cells, const std::string& boundary, bool header, bool index_col,
      int flags = 0) :
        Triangulation(
          read_table<double>(nodes, header, index_col).as_matrix(),
          read_table<int>(cells, header, index_col).as_matrix(),
          read_table<int>(boundary, header, index_col).as_matrix(), flags) { }
    // cubic mesh static constructors (nx, ny, nz: number of nodes along x,y,z axis respectively)
    static Triangulation<3, 3> Parallelepiped(
      double a_x, double b_x, double a_y, double b_y, double a_z, double b_z, int nx, int ny, int nz, int flags = 0) {
        // allocate memory
        Eigen::Matrix<double, Dynamic, Dynamic> nodes(nx * ny * nz, 3);
        Eigen::Matrix<int, Dynamic, Dynamic> cells(
          (nx - 1) * (ny - 1) * (nz - 1) * 6, 4);   // each subcube spliited in 6
        Eigen::Matrix<int, Dynamic, Dynamic> boundary(nx * ny * nz, 1);
        // compute steps along axis
        double h_x = (b_x - a_x) / (nx - 1);
        double h_y = (b_y - a_y) / (ny - 1);
        double h_z = (b_z - a_z) / (nz - 1);
        int cell_id = 0;
        std::array<int, 8> v;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int node_id = k * nx * ny + j * nx + i;
                    nodes.row(node_id) << (a_x + i * h_x), (a_y + j * h_y), (a_z + k * h_z);
                    if (i < nx - 1 && j < ny - 1 && k < nz - 1) {
                        // build vector of vertices of sub-cube
                        int p = k * nx * ny + j * nx + i;
                        v = {p,           p + 1,           p + nx,           p + nx + 1,
                             p + nx * ny, p + nx * ny + 1, p + nx * ny + nx, p + nx * ny + nx + 1};
                        // split sub-cube in 6 tetrahedra (this guarantees a conforming triangulation)
                        cells.row(cell_id + 0) << v[2], v[4], v[6], v[5];
                        cells.row(cell_id + 1) << v[2], v[6], v[5], v[7];
                        cells.row(cell_id + 2) << v[2], v[5], v[7], v[3];
                        cells.row(cell_id + 3) << v[2], v[5], v[3], v[1];
                        cells.row(cell_id + 4) << v[2], v[1], v[5], v[0];
                        cells.row(cell_id + 5) << v[2], v[4], v[5], v[0];
                        cell_id = cell_id + 6;
                    }
                    boundary(node_id, 0) =
                      (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) ? 1 : 0;
                }
            }
        }
        return Triangulation<3, 3>(nodes, cells, boundary, flags);
    }
    static Triangulation<3, 3> Cube(double a, double b, int n_nodes, int flags = 0) {
        return Triangulation<3, 3>::Parallelepiped(a, b, a, b, a, b, n_nodes, n_nodes, n_nodes, flags);
    }
    static Triangulation<3, 3> UnitCube(int n_nodes, int flags = 0) {
        return Triangulation<3, 3>::Cube(0.0, 1.0, n_nodes, flags);
    }
    // getters
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& neighbors() const { return neighbors_; }
    const typename Base::CellType& cell(int id) const {
        if (Base::flags_ & cache_cells) {   // cell caching enabled
            return cell_cache_[id];
        } else {
            cell_ = typename Base::CellType(id, this);
            return cell_;
        }
    }
    bool is_face_on_boundary(int id) const { return boundary_faces_[id]; }
    bool is_edge_on_boundary(int id) const { return boundary_edges_[id]; }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> faces() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          faces_.data(), n_faces_, n_nodes_per_face);
    }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> edges() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          edges_.data(), n_edges_, n_nodes_per_edge);
    }
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& cell_to_faces() const { return cell_to_faces_; }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> face_to_edges() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          face_to_edges_.data(), n_faces_, n_edges_per_face);
    }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> face_to_cells() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          face_to_cells_.data(), n_faces_, 2);
    }
    const std::unordered_map<int, std::unordered_set<int>>& edge_to_cells() const { return edge_to_cells_; }
    const BinaryVector<Dynamic>& boundary_faces() const { return boundary_faces_; }
    int n_faces() const { return n_faces_; }
    int n_edges() const { return n_edges_; }
    int n_boundary_faces() const { return boundary_faces_.count(); }
    int n_boundary_edges() const { return boundary_edges_.count(); }
    // iterators over edges
    class edge_iterator : public internals::filtering_iterator<edge_iterator, EdgeType> {
       protected:
        using Base = internals::filtering_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const Triangulation* mesh_;
        edge_iterator& operator()(int i) {
            Base::val_ = EdgeType(i, mesh_);
            return *this;
        }
       public:
        edge_iterator(int index, const Triangulation* mesh, const BinaryVector<Dynamic>& filter) :
            Base(index, 0, mesh->n_edges_, filter), mesh_(mesh) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        edge_iterator(int index, const Triangulation* mesh) :   // apply no filter
            edge_iterator(index, mesh, BinaryVector<Dynamic>::Ones(mesh->n_edges_)) { }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(n_edges_, this); }
    // iterators over faces
    class face_iterator : public internals::filtering_iterator<face_iterator, FaceType> {
       protected:
        using Base = internals::filtering_iterator<face_iterator, FaceType>;
        using Base::index_;
        friend Base;
        const Triangulation* mesh_;
        int marker_;
        face_iterator& operator()(int i) {
            Base::val_ = FaceType(i, mesh_);
            return *this;
        }
       public:
        using TriangulationType = Triangulation<3, 3>;
        face_iterator(int index, const Triangulation* mesh, const BinaryVector<Dynamic>& filter, int marker) :
            Base(index, 0, mesh->n_faces_, filter), mesh_(mesh), marker_(marker) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        face_iterator(int index, const Triangulation* mesh) :   // apply no filter
	  face_iterator(index, mesh, BinaryVector<Dynamic>::Ones(mesh->n_edges_), Unmarked) { }
        face_iterator(int index, const Triangulation* mesh, int marker) :   // fast construction for end iterators
            Base(index, 0, mesh->n_faces_), marker_(marker) { }
        int marker() const { return marker_; }
    };
    face_iterator faces_begin() const { return face_iterator(0, this); }
    face_iterator faces_end() const { return face_iterator(n_faces_, this); }
    // iterator over boundary faces
    struct boundary_face_iterator : public face_iterator {
        boundary_face_iterator(int index, const Triangulation* mesh) :
            face_iterator(index, mesh, mesh->boundary_faces_, BoundaryAll) { }
        boundary_face_iterator(int index, const Triangulation* mesh, int marker) :   // filter boundary faces by marker
            face_iterator(
              index, mesh,
              marker == BoundaryAll ?
                mesh->boundary_faces_ :
                mesh->boundary_faces_ &
                  make_binary_vector(mesh->faces_markers_.begin(), mesh->faces_markers_.end(), marker),
              marker) { }
    };
    boundary_face_iterator boundary_faces_begin() const { return boundary_face_iterator(0, this); }
    boundary_face_iterator boundary_faces_end() const { return boundary_face_iterator(n_faces_, this); }
    using boundary_iterator = boundary_face_iterator;   // public view of 3D boundary
    BoundaryIterator<Triangulation<3, 3>> boundary_begin(int marker = BoundaryAll) const {
        return BoundaryIterator<Triangulation<3, 3>>(0, this, marker);
    }
    BoundaryIterator<Triangulation<3, 3>> boundary_end(int marker = BoundaryAll) const {
        return BoundaryIterator<Triangulation<3, 3>>(n_faces_, this, marker);
    }
    std::pair<BoundaryIterator<Triangulation<3, 3>>, BoundaryIterator<Triangulation<3, 3>>>
    boundary(int marker = BoundaryAll) const {
        return std::make_pair(boundary_begin(marker), boundary_end(marker));
    }
    const std::vector<int>& faces_markers() const { return faces_markers_; }
    const std::vector<int>& edges_markers() const { return edges_markers_; }
    // set boundary markers
    template <typename Lambda>
    void mark_boundary(int marker, Lambda&& lambda)
        requires(requires(Lambda lambda, FaceType e) {
            { lambda(e) } -> std::same_as<bool>;
        }) {
        fdapde_assert(marker >= 0);
        faces_markers_.resize(n_faces_, Unmarked);
	edges_markers_.resize(n_edges_, Unmarked);
        for (boundary_face_iterator it = boundary_faces_begin(); it != boundary_faces_end(); ++it) {
            if (lambda(*it)) {
                faces_markers_[it->id()] = marker;
                for (int edge_id : it->edge_ids()) { edges_markers_[edge_id] = marker; }
            }
        }
        return;
    }
    template <int Rows, typename XprType> void mark_boundary(const BinMtxBase<Rows, 1, XprType>& mask) {
        fdapde_assert(mask.rows() == n_edges_);
        faces_markers_.resize(n_faces_, 0);
	edges_markers_.resize(n_edges_, 0);
        for (boundary_face_iterator it = boundary_faces_begin(); it != boundary_faces_end(); ++it) {
            if (mask[it->id()]) {
                faces_markers_[it->id()] = 1;
                for (int edge_id : it->edge_ids()) { edges_markers_[edge_id] = 1; }
            }
        }
    }
    template <typename Iterator> void mark_boundary(Iterator first, Iterator last) {
        fdapde_static_assert(
          std::is_convertible_v<typename Iterator::value_type FDAPDE_COMMA int>, INVALID_ITERATOR_RANGE);
        int n_markers = std::distance(first, last);
	bool all_markers_positive = std::all_of(first, last, [](auto marker) { return marker >= 0; });
        fdapde_assert(n_markers == n_faces() && all_markers_positive);
        faces_markers_.resize(n_faces_, Unmarked);
        edges_markers_.resize(n_edges_, Unmarked);
        for (int i = 0; i < n_faces_; ++i) {
            int marker = *(first + i);
            faces_markers_[i] = marker;
            for (int edge_id : face_to_edges().row(i)) { edges_markers_[edge_id] = marker; }
        }
        return;
    }
    // marks all boundary faces
    void mark_boundary(int marker) {
        fdapde_assert(marker >= 0);
        faces_markers_.resize(n_faces_, Unmarked);
        edges_markers_.resize(n_edges_, Unmarked);
        for (auto it = boundary_begin(); it != boundary_end(); ++it) {
            faces_markers_[it->id()] = marker;
            for (int edge_id : it->edge_ids()) { edges_markers_[edge_id] = marker; }
        }
        return;
    }
    void clear_boundary_markers() {
        std::for_each(edges_markers_.begin(), edges_markers_.end(), [](int& marker) { marker = Unmarked; });
        std::for_each(faces_markers_.begin(), faces_markers_.end(), [](int& marker) { marker = Unmarked; });
    }
    // compute the surface triangular mesh of this 3D triangulation
    struct SurfaceReturnType {
        Triangulation<2, 3> triangulation;
        std::unordered_map<int, int> node_map, cell_map;
    };
    SurfaceReturnType surface() const {
        Eigen::Matrix<double, Dynamic, Dynamic> nodes(n_boundary_nodes(), FaceType::n_nodes);
        Eigen::Matrix<int, Dynamic, Dynamic> cells(n_boundary_faces(), FaceType::n_nodes),
          boundary(n_boundary_nodes(), 1);
        std::unordered_map<int, int> node_map;   // bounds node ids in the 3D mesh to rescaled ids on the surface mesh
        std::unordered_map<int, int> cell_map;   // bounds face ids in the 3D mesh to rescaled ids on the surface mesh
        // compute nodes, cells and boundary matrices
        int i = 0, j = 0;
        for (boundary_face_iterator it = boundary_faces_begin(); it != boundary_faces_end(); ++it) {
   	    int cell_id = it->adjacent_cells()[0] != -1 ? it->adjacent_cells()[0] : it->adjacent_cells()[1];
            cell_map[cell_id] = i;
	    Eigen::Matrix<int, Dynamic, 1> node_ids = it->node_ids();
            for (int k = 0; k < FaceType::n_nodes; ++k) {
                int node_id = node_ids[k];
                if (node_map.find(node_id) != node_map.end()) {
                    cells(i, k) = node_map.at(node_id);
                } else {   // never visited face
                    nodes.row(j) = nodes_.row(node_id);
		    boundary(j, 0) = is_node_on_boundary(node_id);
                    cells(i, k) = j;
		    node_map[node_id] = j;
		    j++;
                }
            }
	    i++;
        }
	using internals::map_reverse;
	return {Triangulation<2, 3>(nodes, cells, boundary), map_reverse(node_map), map_reverse(cell_map)};
    }

    // point location
    template <int Rows, int Cols>
    std::conditional_t<Rows == Dynamic || Cols == Dynamic, Eigen::Matrix<int, Dynamic, 1>, int>
    locate(const Eigen::Matrix<double, Rows, Cols>& p) const {
        fdapde_static_assert(
          (Cols == 1 && Rows == embed_dim) || (Cols == Dynamic && Rows == Dynamic),
          YOU_PASSED_A_MATRIX_OF_POINTS_TO_LOCATE_OF_WRONG_DIMENSIONS);
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->locate(p);
    }
    template <typename Derived> Eigen::Matrix<int, Dynamic, 1> locate(const Eigen::Map<Derived>& p) const {
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->locate(p);
    }
    // the set of cells which have node id as vertex
    std::vector<int> node_patch(int id) const {
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->all_locate(Base::node(id));
    }
   protected:
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> neighbors_ {};   // adjacent cells ids (-1: no adjacent cell)
    std::vector<int> faces_, edges_;   // nodes (as row indexes in nodes_ matrix) composing each face and edge
    std::vector<int> face_to_cells_;   // for each face, the ids of adjacent cells
    std::unordered_map<int, std::unordered_set<int>> edge_to_cells_;   // for each edge, the ids of insisting cells
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> cell_to_faces_ {};   // ids of faces composing each cell
    std::vector<int> face_to_edges_;                                           // ids of edges composing each face
    BinaryVector<Dynamic> boundary_faces_ {};   // j-th element is 1 \iff face j is on boundary
    BinaryVector<Dynamic> boundary_edges_ {};   // j-th element is 1 \iff edge j is on boundary
    std::vector<int> faces_markers_;
    std::vector<int> edges_markers_;
    int n_faces_ = 0, n_edges_ = 0;
    mutable std::optional<LocationPolicy> location_policy_ {};
    // cell caching
    std::vector<typename Base::CellType> cell_cache_;
    mutable typename Base::CellType cell_;   // used in case cell caching is off
};

}   // namespace fdapde

#endif   // __FDAPDE_TRIANGULATION_H__
