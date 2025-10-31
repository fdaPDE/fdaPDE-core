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

#ifndef __FDAPDE_R_TREE_H__
#define __FDAPDE_R_TREE_H__

#include "header_check.h"

namespace fdapde {

// R* tree
// Guttman, A. (1984). R-trees: A dynamic index structure for spatial searching. In Proceedings of the 1984 ACM
// SIGMOD international conference on Management of data (pp. 47-57).
// Beckmann, N., Kriegel, H. P., Schneider, R., & Seeger, B. (1990). The R*-tree: An efficient and robust access
// method for points and rectangles. In Proceedings of the 1990 ACM SIGMOD international conference on Management of
// data (pp. 322-331).
  
namespace internals {

struct rtree_quadratic_split {
    rtree_quadratic_split() : M_(0), m_(0) { }
    rtree_quadratic_split(int M, int m) : M_(M), m_(m) { }

    template <typename node_t> std::pair<node_t*, node_t*> apply(node_t* node, int level) {
        using bbox_t = typename node_t::bbox_t;
        using item_t = typename node_t::item_t;
        // allocate memory for new node
        node_t* n = new node_t(M_, level);
        std::vector<int> g;   // ids of items reallocated to n after the split
        Vector<bool, Dynamic> assigned(M_ + 1);
        // pick_seed
        // choose bbox pair with highest covering inefficiency
        double max_d = -std::numeric_limits<double>::infinity();
        int e1 = 0, e2 = 0;
        for (int i = 0; i < M_ + 1; ++i) {
            for (int j = i + 1; j < M_ + 1; ++j) {
                bbox_t tmp = node->item(i).bbox();
                tmp.expand(node->item(j).bbox());   // tmp is the smallest bbox containing both bbox[i] and bbox[j]
                double d = tmp.measure() - node->item(i).bbox().measure() - node->item(j).bbox().measure();
                if (d > max_d) {
                    max_d = d;
                    e1 = i;
                    e2 = j;
                }
            }
        }
        // assign e1 and e2
        assigned[e1] = 1;
        assigned[e2] = 1;
        g.push_back(e2);
        // distribute remaining entries
        std::array<bbox_t, 2> bbox {node->item(e1).bbox(), node->item(e2).bbox()};
        int r = M_ - 1;   // number of not assigned items
        while (std::cmp_less(assigned.count(), M_ + 1)) {
            // if one group has so few entries that all the rest must be assigned to it in order for it to have the
            // minimum number m_, assign them and stop
            bool done = false;
            if (std::cmp_less_equal((M_ + 1 - g.size()) + r, m_)) {
                done = true;
            } else if (std::cmp_less_equal(g.size() + r, m_)) {
                for (int i = 0; i < M_ + 1; ++i) {
                    if (!assigned[i]) { g.push_back(i); }
                }
                done = true;
            }
            if (done) {
                // move data to node n
                for (int i : g) { n->insert(node->item(i)); }
                for (const item_t& item : *n) {
                    node->erase(item, false);
                    if (item.is_node()) { item.node()->set_parent(n); }   // update parent for structural nodes
                }
                node->recompute_bbox();
                return std::make_pair(node, n);
            }
            // pick_next
            // choose next entry to assign
            double max_diff = -std::numeric_limits<double>::infinity();
            int next;
            int b = 0;   // 1 if data has to be moved to n, 0 otherwise
            for (int i = 0; i < M_ + 1; ++i) {
                if (!assigned[i]) {
                    // compute measure increase
                    double d1 = bbox[0].enlargment(node->item(i).bbox());
                    double d2 = bbox[1].enlargment(node->item(i).bbox());
                    double diff = std::abs(d1 - d2);
                    if (diff > max_diff) {
                        max_diff = diff;
                        next = i;
                        b = d1 < d2 || (d1 == d2 && bbox[0].measure() < bbox[1].measure()) ? 0 : 1;
                    }
                }
            }
            if (b) { g.push_back(next); }
            bbox[b].expand(node->item(next).bbox());
            assigned.set(next);
            r--;
        }
        // move data from node to n
        for (int i : g) { n->insert(node->item(i)); }
        for (const item_t& item : *n) {
            node->erase(item, false);
            if (item.is_node()) { item.node()->set_parent(n); }   // update parent for structural nodes
        }
        node->recompute_bbox();
        return std::make_pair(node, n);
    }
   private:
    int M_, m_;
};

struct rtree_star_split {
    rtree_star_split() : M_(0), m_(0) { }
    rtree_star_split(int M, int m) : M_(M), m_(m) { }

    template <typename node_t> std::pair<node_t*, node_t*> apply(node_t* node, int level) {
        constexpr int embed_dim = node_t::embed_dim;
        using bbox_t = typename node_t::bbox_t;
        using item_t = typename node_t::item_t;

        // allocate memory for new node
        node_t* n = new node_t(M_, level);
        // choose split axis
        int best_axis = 0;
        double best_margin = std::numeric_limits<double>::infinity();
        std::vector<int> sorted_ids(node->size());
        std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
        for (int i = 0; i < embed_dim; ++i) {
            // sort entries by lower values of their bounding box, break ties with higher values
            std::sort(sorted_ids.begin(), sorted_ids.end(), [&](int h, int k) {
                double h_min = node->item(h).bbox()[i], k_min = node->item(k).bbox()[i];
                double h_max = node->item(h).bbox()[i + embed_dim], k_max = node->item(k).bbox()[i + embed_dim];
                return h_min < k_min || (h_min == k_min && h_max < k_max);
            });

            // consider all the possible split of M + 1 objects in two groups
            for (int j = m_; j <= M_ - m_ + 1; ++j) {
                // build bounding boxes of the two groups
                bbox_t bbox1 = node->item(sorted_ids[0]).bbox();
                for (int k = 1; k < j; ++k) { bbox1.expand(node->item(sorted_ids[k]).bbox()); }
                bbox_t bbox2 = node->item(sorted_ids[m_]).bbox();
                for (int k = j + 1; k < M_ + 1; ++k) { bbox2.expand(node->item(sorted_ids[k]).bbox()); }

                // compute the sum of margins
                double margin = bbox1.margin() + bbox2.margin();
                if (margin < best_margin) {
                    best_axis = i;
                    best_margin = margin;
                }
            }
        }

        // choose split index
        int idx = 0;
        double best_overlap = std::numeric_limits<double>::infinity();
        double best_measure = std::numeric_limits<double>::infinity();
        std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
        // sort along choosen axis
        std::sort(sorted_ids.begin(), sorted_ids.end(), [&](int h, int k) {
            double h_min = node->item(h).bbox()[best_axis], k_min = node->item(k).bbox()[best_axis];
            double h_max = node->item(h).bbox()[best_axis + embed_dim],
                   k_max = node->item(k).bbox()[best_axis + embed_dim];
            return h_min < k_min || (h_min == k_min && h_max < k_max);
        });

        for (int j = m_; j <= M_ - m_ + 1; ++j) {
            // build bounding boxes of the two groups
            bbox_t bbox1 = node->item(sorted_ids[0]).bbox();
            for (int k = 1; k < j; ++k) { bbox1.expand(node->item(sorted_ids[k]).bbox()); }
            bbox_t bbox2 = node->item(sorted_ids[m_]).bbox();
            for (int k = j + 1; k < M_ + 1; ++k) { bbox2.expand(node->item(sorted_ids[k]).bbox()); }

            // compute the sum of margins
            double measure = bbox1.measure() + bbox2.measure();
            double overlap = bbox1.overlap(bbox2);
            if (overlap < best_overlap || (overlap == best_overlap && measure < best_measure)) {
                idx = j;
                best_overlap = overlap;
                best_measure = measure;
            }
        }

        // split nodes
        for (int i = 0; i < idx; ++i) { n->insert(node->item(sorted_ids[i])); }
        for (const item_t& item : *n) {
            node->erase(item, false);
            if (item.is_node()) { item.node()->set_parent(n); }   // update parent for structural nodes
        }
        node->recompute_bbox();
        return std::make_pair(node, n);
    }
   private:
    int M_, m_;
};

}   // namespace internals

template <int EmbedDim, typename SplitStrategy = internals::rtree_star_split>
class RTree {
   private:
    static constexpr int embed_dim = EmbedDim;

    // forward decl
    struct node_t;
    struct bbox_t;
    struct item_t;

    template <typename SpatialObject_> struct is_valid_spatial_object {
        static constexpr bool value = requires(SpatialObject_ obj) {
            SpatialObject_::embed_dim;
            { obj.bbox() } -> std::convertible_to<std::array<double, 2 * embed_dim>>;
        };
    };
    template <typename SpatialObject_>
    static constexpr bool is_valid_spatial_object_v = is_valid_spatial_object<SpatialObject_>::value;

    // internal structures
    struct bbox_t {
        bbox_t() : bbox_(), measure_(0) { reset(); }
        template <typename T>
            requires(internals::is_vector_like_v<T>)
        explicit bbox_t(const T& bbox) : bbox_(), measure_(1.0) {
            for (int i = 0; i < 2 * embed_dim; ++i) { bbox_[i] = bbox[i]; }
            for (int i = 0; i < embed_dim; ++i) { measure_ *= (bbox_[i + embed_dim] - bbox_[i]); }
        }
        template <typename SpatialObject_>
            requires(requires(SpatialObject_ obj) {
                { obj.bbox() } -> std::same_as<std::array<double, 2 * embed_dim>>;
            })
        explicit bbox_t(const SpatialObject_& obj) : bbox_t(obj.bbox()) { }
        explicit bbox_t(const node_t& node) : bbox_t(node.bbox().coords()) { }
        // copy/move semantic
        bbox_t(const bbox_t& other) : bbox_(other.bbox_), measure_(other.measure_) { }
        bbox_t(bbox_t&& other) : bbox_(std::move(other.bbox_)), measure_(std::exchange(other.measure_, 0.0)) { }
        bbox_t& operator=(const bbox_t& other) {
            bbox_ = other.bbox_;
            measure_ = other.measure_;
            return *this;
        }
        bbox_t& operator=(bbox_t&& other) {
            bbox_ = std::move(other.bbox_);
            measure_ = std::exchange(other.measure_, 0.0);
            return *this;
        }

        // observers
        double measure() const { return measure_; }
        const std::array<double, 2 * embed_dim>& coords() const { return bbox_; }
        double operator[](int i) {
            fdapde_assert(i < 2 * embed_dim);
            return bbox_[i];
        }
        double operator[](int i) const {
            fdapde_assert(i < 2 * embed_dim);
            return bbox_[i];
        }
        // compute measure of the minimal expanded region containing both this and other
        double enlargment(const bbox_t& other) const {
            double exp_measure_ = 1;
            for (int i = 0; i < embed_dim; ++i) {
                exp_measure_ *=
                  (std::max(bbox_[i + embed_dim], other.bbox_[i + embed_dim]) - std::min(bbox_[i], other.bbox_[i]));
            }
            return exp_measure_ - measure_;
        }
        // computes surface measure
        double margin() const {
            constexpr auto pattern = Matrix<int, binomial_coefficient(embed_dim, embed_dim - 1), embed_dim - 1>(
              combinations(embed_dim - 1, embed_dim));
            double margin_ = 0;
            for (int i = 0; i < pattern.rows(); ++i) {
                double tmp = 1;
                for (int j = 0; j < pattern.cols(); ++j) {
                    tmp *= (bbox_[pattern(i, j) + embed_dim] - bbox_[pattern(i, j)]);
                }
                margin_ += 2 * tmp;
            }
            return margin_;
        }
        // compute measure of the intersection region of this and other
        double overlap(const bbox_t& other) const {
            double int_measure_ = 1;
            for (int i = 0; i < embed_dim; ++i) {
                int_measure_ *= (std::max(
                  0.0,
                  std::min(bbox_[i + embed_dim], other.bbox_[i + embed_dim]) - std::max(bbox_[i], other.bbox_[i])));
                if (int_measure_ == 0) return 0;   // early return if no intersection
            }
            return int_measure_;
        }
        // computes the bounding box' center of mass
        std::array<double, embed_dim> centroid() const {
            std::array<double, embed_dim> centroid_;
            for (int i = 0; i < embed_dim; ++i) { centroid_[i] = (bbox_[i + embed_dim] + bbox_[i]) / 2; }
            return centroid_;
        }
        // modifiers
        void expand(const bbox_t& other) {
            for (int i = 0; i < embed_dim; ++i) {
                bbox_[i] = std::min(bbox_[i], other.bbox_[i]);
                bbox_[i + embed_dim] = std::max(bbox_[i + embed_dim], other.bbox_[i + embed_dim]);
            }
            // update measure
            measure_ = 1;
            for (int i = 0; i < embed_dim; ++i) { measure_ *= (bbox_[i + embed_dim] - bbox_[i]); }
        }
        void reset() {
            for (int i = 0; i < embed_dim; ++i) {
                bbox_[i] = std::numeric_limits<double>::infinity();
                bbox_[i + embed_dim] = -std::numeric_limits<double>::infinity();
            }
            measure_ = 0;
        }
        // queries
        template <typename PointT>
            requires(internals::is_vector_like_v<PointT>)
        bool contains(const PointT& p) const {
            fdapde_assert(p.size() == embed_dim);
            for (int i = 0; i < embed_dim; ++i) {
                if (bbox_[i] > p[i] || bbox_[i + embed_dim] < p[i]) return false;
            }
            return true;
        }
        // true if this bbox_t fully contains other
        bool contains(const bbox_t& other) const {
            for (int i = 0; i < embed_dim; ++i) {
                if (bbox_[i] > other.bbox_[i] || bbox_[i + embed_dim] < other.bbox_[i + embed_dim]) return false;
            }
            return true;
        }
        // true if this bbox_t partially overlapls with other
        template <typename VectorT>
            requires(internals::is_vector_like_v<VectorT>)
        bool intersects(const VectorT& vec) const {
            fdapde_assert(vec.size() == 2 * embed_dim);
            for (int i = 0; i < embed_dim; ++i) {
                if (bbox_[i] > vec[i + embed_dim] || bbox_[i + embed_dim] < vec[i]) return false;
            }
            return true;
        }
        bool intersects(const bbox_t& other) const {
            for (int i = 0; i < embed_dim; ++i) {
                if (bbox_[i] > other.bbox_[i + embed_dim] || bbox_[i + embed_dim] < other.bbox_[i]) return false;
            }
            return true;
        }
       private:
        std::array<double, 2 * embed_dim> bbox_;   // (x1_min, x2_min, ..., xn_min, x1_max, ..., xn_max)
        double measure_;
    };
    // data item stored in node
    struct item_t {
       private:
        enum class type_t {
            NODE,
            DATA
        };
        type_t type_;
        union {
            node_t* node_;   // internal nodes: pointer to child
            int data_;       // leaf nodes: database index at which this spatial object is stored
        };
        bbox_t bbox_;   // smallest rectangle that spatially contains this item
       public:
        // constructor
        item_t() = default;
        template <typename BBoxT>
            requires(std::is_convertible_v<BBoxT, bbox_t>)
        explicit item_t(int index, const BBoxT& bbox) : type_(type_t::DATA), data_(index), bbox_(bbox) { }
        explicit item_t(node_t* node) : type_(type_t::NODE), node_(node), bbox_(node_->bbox()) { }
        // copy/move semantic
        item_t(const item_t& other) = default;
        item_t(item_t&& other) = default;
        item_t& operator=(const item_t& other) = default;
        item_t& operator=(item_t&& other) = default;

        // observers
        const bbox_t& bbox() const { return bbox_; }
        node_t* node() const {
            fdapde_assert(type_ == type_t::NODE);
            return node_;
        }
        int data() const {
            fdapde_assert(type_ == type_t::DATA);
            return data_;
        }
        bool is_data() const { return type_ == type_t::DATA; }
        bool is_node() const { return type_ == type_t::NODE; }
        // comparison
        friend bool operator==(const item_t& lhs, const item_t& rhs) {
            if (lhs.type_ != rhs.type_) return false;
            return lhs.type_ == type_t::NODE ? lhs.node_ == rhs.node_ : lhs.data_ == rhs.data_;
        }
        friend bool operator!=(const item_t& lhs, const item_t& rhs) { return !(lhs == rhs); }
        // modifiers
        bbox_t& bbox() { return bbox_; }
        void set_node(node_t* node) {
            fdapde_assert(type_ == type_t::NODE);
            node_ = node;
        }
        // shallow destruction (not-owned memory)
        ~item_t() = default;
    };
    // structural tree node
    struct node_t {
        using bbox_t = typename RTree<EmbedDim, SplitStrategy>::bbox_t;
        using item_t = typename RTree<EmbedDim, SplitStrategy>::item_t;
        static constexpr int embed_dim = EmbedDim;

        node_t(int M, int level) : size_(0), capacity_(M + 1), level_(level), bbox_() {
            data_.resize(M + 1);   // allow overflow to use this buffer for in-place split logic
            free_.resize(M + 1, true);
            map_ .resize(M + 1, 0);
        }
        // observers
        node_t* parent() const { return parent_; }
        bool is_leaf() const { return level_ == 0; }
        bool is_internal() const { return level_ > 0; }
        bool is_root() const { return parent_ == nullptr; }
        std::size_t size() const { return size_; }
        const bbox_t& bbox() const { return bbox_; }
        int level() const { return level_; }
        const item_t& item(int i) const {
            fdapde_assert(i < size_);
            return data_[map_[i]];
        }
        // modifiers
        bbox_t& bbox() { return bbox_; }
        item_t& item(int i) {
            fdapde_assert(i < size_);
            return data_[map_[i]];
        }
        void recompute_bbox() {
            bbox_.reset();
            for (const item_t& item : *this) { bbox_.expand(item.bbox()); }
        }
        void update_bbox_of(const item_t& n) {
            int i = 0;
            for (const item_t& item : *this) {
                if (item.node() == n.node()) { break; }
                i++;
            }
            data_[map_[i]].bbox() = n.bbox();
        }
        template <typename ItemT> void insert(ItemT&& item, bool update_bbox = true) {
            fdapde_assert(size_ < capacity_);
            // find idx of first free slot in data_
            int j = 0;
            for (int i = 0, n = capacity_; i < n; ++i) {
                if (free_[i]) {
                    j = i;
                    break;
                }
            }
            data_[j] = item;
            free_[j] = false;
            map_[size_] = j;
            size_++;
            if (update_bbox) bbox_.expand(item.bbox());
        }
        void erase(const item_t& item, bool update_bbox = true) {
            // search physical index of item
            int i = 0, k = map_[0];
            for (; i < size_; ++i, k = map_[i]) {
                if (item == data_[k]) break;
            }
            free_[k] = true;
            size_--;
            for (int j = i; j < size_; ++j) { map_[j] = map_[j + 1]; }   // realign logical-physical mapping
            if (update_bbox) recompute_bbox();
        }
        void set_parent(node_t* parent) { parent_ = parent; }
        // iterators
        struct iterator {
            using value_type = item_t;
            using pointer = std::add_pointer_t<value_type>;
            using reference = std::add_lvalue_reference_t<value_type>;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::forward_iterator_tag;

            iterator() : node_(nullptr), index_(0) { }
            iterator(node_t* node, int index) : node_(node), index_(index) { }
            // increment
            iterator& operator++() {
                index_++;
                return *this;
            }
            // accessors
            const item_t& operator*() const { return node_->item(index_); }
            item_t& operator*() { return node_->item(index_); }
            const item_t* operator->() const { return std::addressof(node_->item(index_)); }
            item_t* operator->() { return std::addressof(node_->item(index_)); }
            // comparison
            friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
           private:
            node_t* node_;
            int index_;
        };
        iterator begin() { return iterator(this, 0); }
        iterator end() { return iterator(this, size_); }
       private:
        // tree structure data
        node_t* parent_ = nullptr;
        std::vector<item_t> data_;
        std::vector<int> map_;     // map_[i]: physical index of i-th valid item, for i < size_
        std::vector<bool> free_;   // free_[i] == true \iff i-th physical slot available for writing
        int size_ = 0, capacity_ = 0;
        int level_;     // node's level (NB: leafs are at level 0)
        bbox_t bbox_;   // minimal bounding box enclosing all bounding boxes rooted at this node
    };

    // insert internal utilities

    // select a node at depth target_level in which to place spatial object obj
    node_t* choose_subtree_(const bbox_t& obj_bbox, int target_level) {
        node_t* curr = root_;

        while (curr->level() != target_level) {
            node_t* best = nullptr;

            if (target_level == 0 && curr->level() == 1) {   // use overlap heuristic for leaf insertion
                struct overlap_t {
                    node_t* node;
                    bbox_t bbox;
                    double enlargement;
                };
                std::vector<overlap_t> overlaps(curr->size());
                double best_enlargement = std::numeric_limits<double>::infinity();

                // compute enlargement of all items
                for (int i = 0, n = curr->size(); i < n; ++i) {
                    overlaps[i].node = curr->item(i).node();
                    overlaps[i].bbox = curr->item(i).bbox();
                    overlaps[i].bbox.expand(obj_bbox);
                    overlaps[i].enlargement = overlaps[i].bbox.measure() - overlaps[i].node->bbox().measure();
                    if (
                      overlaps[i].enlargement < best_enlargement ||
                      (overlaps[i].enlargement == best_enlargement &&
                       overlaps[i].node->bbox().measure() < best->bbox().measure())) {
                        best_enlargement = overlaps[i].enlargement;
                        best = curr->item(i).node();
                    }
                }
                // near minimum overlap heuristic
                if (curr->size() > overlap_factor_) {
                    // sort objects in increasing order of enlargment
                    std::sort(overlaps.begin(), overlaps.end(), [&](const overlap_t& a, const overlap_t& b) {
                        return a.enlargement < b.enlargement;
                    });
                }
                int n = fdapde::min(overlap_factor_, curr->size());
                double best_overlap = std::numeric_limits<double>::infinity();
                // compute overlap of top n entries
                for (int i = 0; i < n; ++i) {
                    double overlap = 0;
                    for (int j = 0, m = curr->size(); j < m; ++j) {
                        if (i != j) {
                            double tmp = overlaps[i].bbox.overlap(curr->item(j).bbox());
                            // if enlarged bbox doesn't intersect j-th bbox, neither the old one can
                            if (!almost_zero(tmp)) {
                                overlap += tmp - overlaps[i].node->bbox().overlap(curr->item(j).bbox());
                            }
                        }
                    }
                    if (overlap < best_overlap) {
                        best_overlap = overlap;
                        best = overlaps[i].node;
                    } else {
                        // break ties
                        if (almost_equal(overlap, best_overlap)) {
                            if (
                              overlaps[i].enlargement < best_enlargement ||
                              (overlaps[i].enlargement == best_enlargement &&
                               overlaps[i].node->bbox().measure() < best->bbox().measure())) {
                                best_overlap = overlap;
                                best_enlargement = overlaps[i].enlargement;
                                best = overlaps[i].node;
                            }
                        }
                    }
                }
            } else {
                // choose the entry whose bounding box needs least measure enlargement to include obj
                double best_enlargement = std::numeric_limits<double>::infinity();
                for (int i = 0, m = curr->size(); i < m; ++i) {
                    double enlargement = curr->item(i).bbox().enlargment(obj_bbox);
                    if (
                      enlargement < best_enlargement ||
                      (enlargement == best_enlargement && best->bbox().measure() < curr->item(i).bbox().measure())) {
                        best = curr->item(i).node();
                        best_enlargement = enlargement;
                    }
                }
            }
            curr = best;
        }
        return curr;
    }
    node_t* choose_leaf_(const bbox_t& obj_bbox) { return choose_subtree_(obj_bbox, 0); }
    // selects the most (euclidean) distant reinsert_factor_ * M_ entries from n's centroid, and perform a forced insert
    void reinsert_(node_t* n, std::vector<bool>& ctx) {
        using point_t = std::array<double, embed_dim>;
        struct reinsert_entry {
            item_t item;       // structural node or actual spatial data object to reinsert
            double distance;   // distance of this node from n's centroid

            reinsert_entry(item_t item_, double distance_) : item(std::move(item_)), distance(distance_) { }
        };
        std::vector<reinsert_entry> aux;
        aux.reserve(n->size());
        auto euclidean_distance = [](const point_t& a, const point_t& b) -> double {
            double dist = 0.0;
            for (int i = 0; i < embed_dim; ++i) { dist += (b[i] - a[i]) * (b[i] - a[i]); }
            return dist;
        };
        // compute distances between the center of n's bounding box and the ones of all its items
        const point_t p = n->bbox().centroid();
        for (int i = 0, m = n->size(); i < m; ++i) {
            const item_t& item = n->item(i);
            double dist = euclidean_distance(p, item.bbox().centroid());
            aux.emplace_back(item, dist);
        }
        // sort entries in decreasing order of their distance (we want to reinsert more distant nodes)
        std::sort(aux.begin(), aux.end(), [](const auto& a, const auto& b) { return a.distance > b.distance; });

        // remove first q entries
        int q = static_cast<int>(reinsert_factor_ * M_);
        std::vector<item_t> to_reinsert;
        to_reinsert.reserve(q);
        for (int i = 0; i < q; ++i) {
            n->erase(aux[i].item, false);
            to_reinsert.push_back(std::move(aux[i].item));
        }
        n->recompute_bbox();   // adjust n's bounding box

        // reinsert entries
        for (item_t& it : to_reinsert) {
            node_t* target = nullptr;
            if (n->is_leaf()) {
                target = choose_subtree_(it.bbox(), n->level());
            } else {
                target = choose_subtree_(it.node()->bbox(), n->level());
            }
            target->insert(it);
            adjust_tree_(target, ctx);
        }
        return;
    }
    // ascend from a node n to the root, adjusting covering rectangles and propagating node splits and reinsertion
    void adjust_tree_(node_t* n, std::vector<bool>& ctx) {
        node_t *n1 = n, *n2 = nullptr;

        while (true) {
            node_t* parent = n1->parent();
            // handle overlfow occurred
            if (n1->size() > M_) {
                if (!n1->is_root() && !ctx[n1->level()]) {
                    ctx[n1->level()] = true;
                    reinsert_(n1, ctx);
                    parent->update_bbox_of(item_t(n1));
                    break;   // reinsertion stop propagation
                } else {
                    const auto& [e1, e2] = split_.apply(n1, n1->level());
                    n1 = e1;
                    n2 = e2;
                    if (parent) {
                        parent->update_bbox_of(item_t(n1));
                        // just insert n2, as n1 is modified in-place
                        parent->insert(item_t(n2));
                        n2->set_parent(parent);
                    }
                }
            }
            if (!parent) break;
            // bounding box update
            if (parent->size() < M_ + 1) { parent->recompute_bbox(); }
            // go one level up
            n1 = parent;
            n2 = nullptr;
        }
        // if root was splitted, create new root
        if (n2 != nullptr) {
            node_t* new_root = new node_t(M_, ++depth_);
            n1->set_parent(new_root);
            n2->set_parent(new_root);
            new_root->insert(item_t(n1));
            new_root->insert(item_t(n2));
            root_ = new_root;
        }
        return;
    }

    // erase internal utilities

    // find leaf node containing obj. returns nullptr if no obj is found
    std::pair<node_t*, const item_t*> find_leaf_(const bbox_t& obj_bbox, int index) {
        node_t* curr;
        std::stack<node_t*> stack_;
        stack_.push(root_);
        while (!stack_.empty()) {
            curr = stack_.top();
            stack_.pop();
            if (!curr->is_leaf()) {
                for (const auto& item : *curr) {
                    if (obj_bbox.intersects(item.bbox())) { stack_.push(item.node()); }
                }
            } else {
                // search for exact match in leaf
                for (const auto& item : *curr) {
                    if (item.data() == index) { return std::make_pair(curr, std::addressof(item)); }
                }
            }
        }
        return std::make_pair(nullptr, nullptr);
    }
    // given a leaf node l from whch an entry has been deleted, eliminate it if has too few entries and relocate.
    // Propagate node elimination upward, adjusting covering rectangles as necessary
    void condense_tree_(node_t* l) {
        std::vector<std::pair<int, bbox_t>> Q;   // set of eliminated items
        std::vector<node_t*> killed;
        node_t* n = l;
        while (!n->is_root()) {
            node_t* parent = n->parent();
            if (n->size() < m_) {   // too few entries, eliminate node
                if (n->is_leaf()) {
                    for (const item_t& item : *n) { Q.emplace_back(item.data(), item.bbox()); }
                    killed.push_back(n);
                } else {
                    // if node was not a leaf, eliminate its entire subtree
                    std::stack<node_t*> stack_;
                    stack_.push(n);
                    node_t* curr;
                    while (!stack_.empty()) {
                        curr = stack_.top();
                        stack_.pop();
                        if (curr->is_leaf()) {
                            for (const item_t& item : *curr) { Q.emplace_back(item.data(), item.bbox()); }
                        } else {
                            for (const item_t& item : *curr) { stack_.push(item.node()); }
                        }
                        killed.push_back(curr);
                    }
                }
                parent->erase(item_t(n), false);
            } else {
                parent->update_bbox_of(item_t(n));
            }
            parent->recompute_bbox();
            n = parent;
        }
        // free memory and insert
        for (node_t* node : killed) { delete node; }
        for (const auto& [obj_data, obj_bbox] : Q) { insert(obj_data, obj_bbox); }
        return;
    }

    // internals

    // perform a depth-first visit of the tree, executing Functor at each node
    template <typename Functor>
        requires(requires(Functor f, node_t* n) {
            { f(n) } -> std::same_as<void>;
        })
    void dfs_visit_(Functor&& f) const {
        node_t* curr;
        std::stack<node_t*> stack_;
        stack_.push(root_);
        while (!stack_.empty()) {
            curr = stack_.top();
            stack_.pop();
            if (!curr->is_leaf()) {
                for (const auto& item : *curr) { stack_.push(item.node()); }
            }
            f(curr);   // functor execution
        }
        return;
    }
    // deep copies other into this
    void clone_(const RTree& other) {
        // copy basic data members
        M_ = other.M_;
        m_ = other.m_;
        depth_ = other.depth_;
        split_ = other.split_;
        overlap_factor_ = other.overlap_factor_;
        reinsert_factor_ = other.reinsert_factor_;

        if (!other.root_) {
            root_ = nullptr;
            return;
        }
        // deep copy tree content (bfs visit)
        root_ = new node_t(*other.root_);
        std::queue<std::pair<node_t*, node_t*>> queue_;
        queue_.push({root_, other.root_});
        while (!queue_.empty()) {
            auto [dst, src] = queue_.front();
            queue_.pop();
            for (int i = 0, n = src->size(); i < n; ++i) {
                node_t* src_child = src->item(i).node();
                node_t* dst_child = new node_t(*src_child);   // child shallow copy

                // update parent-child pointer relationship
                dst_child->set_parent(dst);
                dst->item(i).set_node(dst_child);
                // push child
                if (!src_child->is_leaf()) { queue_.push({dst_child, src_child}); }
            }
        }
        return;
    }
    // moves other into this, leavs other in an empty state
    void move_(RTree&& other) {
        // copy basic data members
        M_ = other.M_;
        m_ = other.m_;
        depth_ = other.depth_;
        split_ = other.split_;
        overlap_factor_ = other.overlap_factor_;
        reinsert_factor_ = other.reinsert_factor_;

        // move tree resources
        root_ = std::exchange(other.root_, nullptr);
        return;
    }  

    // bulk loading
    // Leutenegger, S. T., Lopez, M. A., & Edgington, J. (1997, April). STR: A simple and efficient algorithm for R-tree
    // packing. In Proceedings 13th international conference on data engineering (pp. 497-506). IEEE

    // pack data in (data.size() + M_ - 1) / M_ leaves
    std::vector<node_t*> str_make_leaves_(const std::vector<std::pair<int, bbox_t>>& data) {
        using point_t = std::array<double, embed_dim>;
        int n = data.size();
        std::vector<int> idxs(n);
        std::iota(idxs.begin(), idxs.end(), 0);
        // pre-compute bounding boxes centroid
        std::vector<point_t> centroid;
        centroid.reserve(n);
        for (int i = 0; i < n; ++i) { centroid.push_back(data[i].second.centroid()); }

        std::vector<node_t*> leaves;
	leaves.reserve((n + M_ - 1) / M_);
        // allocate space for leaf, populate it with data in range [begin, end)
        auto make_leaf_ = [&](int begin, int end) {
            node_t* leaf = new node_t(M_, 0);
            for (int i = begin; i < end; ++i) {
                const auto& [index, bbox] = data[idxs[i]];
                leaf->insert(item_t(index, bbox), false);
            }
	    leaf->recompute_bbox();
            leaves.push_back(leaf);
        };
        struct slice_t {
            int dim;          // splitting dimension
            int begin, end;   // start - end index in idxs vec
        };
        std::vector<slice_t> stack_;

	// start recursion
	stack_.push_back({0, 0, n});
        while (!stack_.empty()) {
            slice_t slice = stack_.back();
            stack_.pop_back();

            int count = slice.end - slice.begin;
            if (count <= M_) {   // less than M_ items in slice, create leaf
                make_leaf_(slice.begin, slice.end);
                continue;
            }
            if (slice.dim == embed_dim - 1) {
                // last dimension, chop slice into groups of M_ items and make leaves
                int curr = slice.begin;
                while (curr < slice.end) {
                    int chunk_end = std::min(curr + M_, slice.end);
                    make_leaf_(curr, chunk_end);
                    curr = chunk_end;
                }
                continue;
            }
            // general case
            int n_slices = std::ceil((double)(count + M_ - 1) / M_);
            int S = fdapde::max(1, std::ceil(std::pow(n_slices, 1.0 / (embed_dim - slice.dim))));   // axis slice factor

            // sort along current dimension
            std::sort(idxs.begin() + slice.begin, idxs.begin() + slice.end, [&](int a, int b) {
                double ca = centroid[a][slice.dim];
                double cb = centroid[b][slice.dim];
                if (ca != cb) return ca < cb;
                const auto& [ia, ba] = data[a];
                const auto& [ib, bb] = data[b];
                if (ba[slice.dim] != bb[slice.dim]) return ba[slice.dim] < bb[slice.dim];
                return a < b;
            });

            // prepare for next dimension
            int base = count / S;
	    int res  = count % S;
            int i = slice.begin;
            for (int s = 0; s < S; ++s) {
                int sz = std::min(base + (s < res ? 1 : 0), slice.end - i);
                stack_.push_back({slice.dim + 1, i, i + sz});
                i += sz;
            }
        }
        return leaves;
    }

    // build the R-Tree structure starting from the zero-level leaves. returns the root
    node_t* str_make_tree_(const std::vector<node_t*>& leaves) {
        if (leaves.empty()) return nullptr;
        if (leaves.size() == 1) {
            depth_ = 0;
            return leaves[0];
        }
        int dim = 0;
        std::vector<node_t*> current = leaves;
        std::vector<node_t*> next;
        depth_ = 0;

        while (current.size() > 1) {
            next.clear();
            next.reserve((current.size() + M_ - 1) / M_);
            dim = (dim + 1) % embed_dim;   // rotating splitting dimension (STR block sorting logic)

            // sort bounding boxes along current split dimension
            std::sort(current.begin(), current.end(), [dim](node_t* a, node_t* b) {
                double ca = 0.5 * (a->bbox()[dim] + a->bbox()[dim + embed_dim]);
                double cb = 0.5 * (b->bbox()[dim] + b->bbox()[dim + embed_dim]);
                if (ca != cb) return ca < cb;
                return a->bbox()[dim] < b->bbox()[dim];
            });

            // pack groups of up to M_ children into parent nodes
            for (int i = 0, n = current.size(); i < n; i += M_) {
                node_t* node = new node_t(M_, depth_ + 1);
                // push items in node
                int end = fdapde::min(i + M_, current.size());
                for (int j = i; j < end; ++j) {
                    node->insert(item_t(current[j]), false);
                    current[j]->set_parent(node);
                }
                node->recompute_bbox();
                next.push_back(node);
            }
	    current.swap(next);
	    depth_++;
        }
        return current[0];
    }

    node_t* root_ = nullptr;
    int M_;       // maximum number of entries per node
    int m_;       // minimum number of entries per node
    int depth_;   // current tree depth

    SplitStrategy split_ {};
    int overlap_factor_;       // number of items considered in the ovrelap heuristic
    double reinsert_factor_;   // proportion of reinserted items in case of node overflow
   public:
    RTree(int M, int m, int overlap_factor, double reinsert_factor) :
        M_(M), m_(m), depth_(0), split_(M, m), overlap_factor_(overlap_factor), reinsert_factor_(reinsert_factor) {
        fdapde_assert(m_ >= 2 && m_ <= M / 2);
        root_ = new node_t(M_, 0);
    }
    RTree(int M, int m) :
        M_(M), m_(m), depth_(0), split_(M, m), overlap_factor_(3.0 / 4.0 * M_), reinsert_factor_(0.4) {
        fdapde_assert(m_ >= 2 && m_ <= M / 2);
        root_ = new node_t(M_, 0);
    }
    RTree(int M) :
        M_(M), m_(M / 2), depth_(0), split_(M, M / 2), overlap_factor_(3.0 / 4.0 * M_), reinsert_factor_(0.4) {
        root_ = new node_t(M_, 0);
    }
    RTree() : RTree(32, 16) { }
    // copy/move semantic
    RTree(const RTree& other) : root_(nullptr) { clone_(other); }
    RTree& operator=(const RTree& other) {
        if (this != std::addressof(other)) {
            clear();   // free old memory
            clone_(other);
        }
        return *this;
    }
    RTree(RTree&& other) { move_(std::forward<RTree&&>(other)); }
    RTree& operator=(RTree&& other) {
        if (this != std::addressof(other)) { move_(std::forward<RTree&&>(other)); }
        return *this;
    }

    // modifiers
    template <typename SpatialObject_>
        requires(is_valid_spatial_object_v<SpatialObject_>)
    void insert(const SpatialObject_& obj, int index) {
        fdapde_static_assert(SpatialObject_::embed_dim == embed_dim, INCORRECT_SPATIAL_OBJECT_EMBEDDING_DIMENSION);
        bbox_t obj_bbox(obj);
        // select a leaf where insert obj
        node_t* leaf_node = choose_leaf_(obj_bbox);
        leaf_node->insert(item_t(index, obj_bbox));
        // propagate tree rebalancing up
        std::vector<bool> ctx(depth_, false);
        adjust_tree_(leaf_node, ctx);
        return;
    }
    template <typename SpatialObject_>
        requires(is_valid_spatial_object_v<SpatialObject_>)
    void erase(const SpatialObject_& obj, int index) {
        // find leaf containing obj
        const auto& [leaf_node, item] = find_leaf_(obj.bbox(), index);
        if (!leaf_node) return;
        leaf_node->erase(*item);
        condense_tree_(leaf_node);
        // if the root node has only one child after the condensation, make the child the new root
        if (!root_->is_leaf() && root_->size() == 1) {
            node_t* new_root = root_->item(0).node();
            new_root->set_parent(nullptr);
            delete root_;
            root_ = new_root;
            depth_--;
        }
        return;
    }
    // STR bulk loading
    template <typename SpatialIndex>
        requires(
          internals::is_vector_like_v<SpatialIndex> &&
          requires(internals::subscript_result_of_t<SpatialIndex, int> record) {
              { record.bbox() } -> std::convertible_to<std::array<double, 2 * embed_dim>>;
          })
    void bulk_load(const SpatialIndex& data) {
        fdapde_assert(data.size() > 0);
        int n = data.size();
        // compute bounding boxes of data
        std::vector<std::pair<int, bbox_t>> data_;
        data_.reserve(n);
        for (int i = 0; i < n; ++i) { data_.emplace_back(i, data[i].bbox()); }
        // build leaves via STR tiling
        std::vector<node_t*> leaves = str_make_leaves_(data_);
	// build overall tree structure
        root_ = str_make_tree_(leaves);
        return;
    }

    // observers
    node_t* root() { return root_; }
    int depth() const { return depth_; }
    int size() const {
        int size_ = 0;
        dfs_visit_([&](node_t* curr) {
            if (curr->is_leaf()) size_ += curr->size();
        });
        return size_;
    }

    // geometric queries

    // find all candidate spatial objects which may intersect with query
    template <typename QueryObject>
        requires(std::is_convertible_v<QueryObject, std::array<double, 2 * embed_dim>>)
    std::vector<int> intersect_query(const QueryObject& query) const {
        std::vector<int> candidates;
        bbox_t query_bbox(query);
        node_t* curr;
        std::stack<node_t*> stack_;
        stack_.push(root_);
        while (!stack_.empty()) {
            curr = stack_.top();
            stack_.pop();
            // perform intersection test
            if (curr->is_leaf()) {
                for (const auto& item : *curr) {
                    if (item.bbox().intersects(query_bbox)) { candidates.push_back(item.data()); }
                }
            } else {
                for (const auto& item : *curr) {
                    if (item.bbox().intersects(query_bbox)) { stack_.push(item.node()); }
                }
            }
        }
        return candidates;
    }
    // finds all candidate spatial objects which may contain point
    template <typename PointT>
        requires(internals::is_vector_like_v<PointT>)
    std::vector<int> locate_query(PointT&& point) const {
        fdapde_assert(point.size() == embed_dim);
        std::vector<int> candidates;
        node_t* curr;
        std::stack<node_t*> stack_;
        stack_.push(root_);

        while (!stack_.empty()) {
            curr = stack_.top();
            stack_.pop();
            // the bounding box doesn't contains point, skip entire subtree
            if (!curr->bbox().contains(point)) continue;

            if (curr->is_leaf()) {
                for (const auto& item : *curr) {
                    if (item.bbox().contains(point)) { candidates.push_back(item.data()); }
                }
            } else {
                for (const auto& item : *curr) {
                    if (item.bbox().contains(point)) { stack_.push(item.node()); }
                }
            }
        }
        return candidates;
    }

    // dfs memory deallocation
    void clear() {
        if (root_) dfs_visit_([](node_t* n) { delete n; });
        root_ = nullptr;
    }
    ~RTree() { clear(); }
};

}   // namespace fdapde

#endif   // __FDAPDE_R_TREE_H__
