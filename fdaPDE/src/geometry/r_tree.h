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

#ifndef __FDAPDE_R_TREE_2_H__
#define __FDAPDE_R_TREE_2_H__

#include "header_check.h"

namespace fdapde {

namespace internals {

struct rtree_quadratic_split {
    rtree_quadratic_split() : M_(0), m_(0) { }
    rtree_quadratic_split(int M, int m) : M_(M), m_(m) { }

    template <typename node_t> std::pair<node_t*, node_t*> apply(node_t* node, int level) {
        using bbox_t = typename node_t::bbox_t;
	using item_t = typename node_t::item_t;
	constexpr int embed_dim = node_t::embed_dim;
	
        // allocate memory for new node
        node_t* n = new node_t(M_, level);
        std::vector<int> g;   // ids of items reallocated to n after the split
        BinaryVector<Dynamic> assigned(M_ + 1);
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
        assigned.set({e1, e2});
        g.push_back(e2);
        // distribute remaining entries
        std::array<bbox_t, 2> bbox {node->item(e1).bbox(), node->item(e2).bbox()};
        int r = M_ - 1;   // number of not assigned items
        while (std::cmp_less(assigned.count(), M_ + 1)) {
            // if one group has so few entries that all the rest must be assigned to it in order for it to have the
            // minimum number m_, assign them and stop
            bool done = false;
            if ((M_ + 1 - g.size()) + r <= m_) {
                done = true;
            } else if (g.size() + r <= m_) {
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

// R* tree
// Guttman, A. (1984). R-trees: A dynamic index structure for spatial searching. In Proceedings of the 1984 ACM
// SIGMOD international conference on Management of data (pp. 47-57).
// Beckmann, N., Kriegel, H. P., Schneider, R., & Seeger, B. (1990). The R*-tree: An efficient and robust access
// method for points and rectangles. In Proceedings of the 1990 ACM SIGMOD international conference on Management of
// data (pp. 322-331).
template <typename SpatialObject, typename SplitStrategy = internals::rtree_star_split>
    requires(requires(SpatialObject obj) {
        SpatialObject::embed_dim;
        { obj.bbox() } -> std::convertible_to<std::array<double, 2 * SpatialObject::embed_dim>>;
    })
class RTree2 {
   private:
    static constexpr int embed_dim = SpatialObject::embed_dim;

    // forward decl
    struct node_t;
    struct bbox_t;
    struct item_t;

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
        // observers
        double measure() const { return measure_; }
        const std::array<double, 2 * embed_dim>& coords() const { return bbox_; }
        double operator[](int i) {
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
        enum class type_t { DATA, NODE };
       private:
        union {   // child node or actual data
            node_t* node_;
            const SpatialObject* data_;
        };
        bbox_t bbox_;   // smallest rectangle that spatially contains object pointed by child
        type_t type_;
       public:
        // constructor
        item_t() = default;
        explicit item_t(const SpatialObject& data) :
            data_(std::addressof(data)), type_(type_t::DATA), bbox_(data_->bbox()) { }
        explicit item_t(node_t* node) : node_(node), type_(type_t::NODE), bbox_(node_->bbox()) { }
        // observers
        const bbox_t& bbox() const { return bbox_; }
        node_t* node() const {
            fdapde_assert(type_ == type_t::NODE);
            return node_;
        }
        const SpatialObject& data() const {
            fdapde_assert(type_ == type_t::DATA);
            return *data_;
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
        // shallow destruction (not-owned memory)
        ~item_t() = default;
    };
    // structural tree node
    struct node_t {
        using bbox_t = typename RTree2<SpatialObject, SplitStrategy>::bbox_t;
        using item_t = typename RTree2<SpatialObject, SplitStrategy>::item_t;
        static constexpr int embed_dim = RTree2<SpatialObject, SplitStrategy>::embed_dim;

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
            const iterator* operator->() const { return std::addressof(node_->item(index_)); }
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

    // select a leaf node in which to place a new index entry obj
    template <typename SpatialObject_> node_t* choose_subtree_(const SpatialObject_& obj, int target_level) {
        bbox_t obj_bbox(obj);
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
                if (curr->size() > m_overlap_factor_) {
                    // sort objects in increasing order of enlargment
                    std::sort(overlaps.begin(), overlaps.end(), [&](const overlap_t& a, const overlap_t& b) {
                        return a.enlargement < b.enlargement;
                    });
                }
                int n = fdapde::min(m_overlap_factor_, curr->size());
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
    node_t* choose_leaf_(const SpatialObject& obj) { return choose_subtree_(obj, 0); }

    // find the leaf node containing the index entry obj, togheter with its position. returns nullptr if no obj found
    std::pair<node_t*, const item_t*> find_leaf_(const SpatialObject& obj) {
        bbox_t obj_bbox(obj);
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
                    if (std::addressof(item.data()) == std::addressof(obj)) {
                        return std::make_pair(curr, std::addressof(item));
                    }
                }
            }
        }
        return std::make_pair(nullptr, nullptr);
    }

    // given a leaf node l from whch an entry has been deleted, eliminate it if has too few entries and relocate.
    // Propagate node elimination upward, adjusting covering rectangles as necessary
    void condense_tree_(node_t* l) {
        std::vector<const SpatialObject*> Q;   // set of eliminated items
	std::vector<node_t*> killed; 
        node_t* n = l;
        while (!n->is_root()) {
            node_t* parent = n->parent();
            if (n->size() < m_) {   // too few entries, eliminate node
                if (n->is_leaf()) {
                    for (const item_t& item : *n) { Q.push_back(std::addressof(item.data())); }
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
                            for (const item_t& item : *curr) { Q.push_back(std::addressof(item.data())); }
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
        for (const SpatialObject* obj : Q) { insert(*obj); }
        return;
    }

    // reinsertion logic
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
	      target = choose_leaf_(it.data());
            } else {
	      target = choose_subtree_(*it.node(), n->level());
            }
	    target->insert(it);
            adjust_tree_(target, ctx);
        }
        return;
    }

    // ascend from a node n to the root, adjusting covering rectangles and propagating node splits and reinsertion as
    // necessary
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

    node_t* root_ = nullptr;
    int M_;       // maximum number of entries per node
    int m_;       // minimum number of entries per node
    int depth_;   // current tree depth

    SplitStrategy split_ {};
    int m_overlap_factor_ = 3 / 4 * M_;
    double reinsert_factor_ = 0.4;
   public:
    RTree2(int M, int m) : M_(M), m_(m), depth_(0), split_(M, m) {
        fdapde_assert(m_ >= 2 && m_ <= M / 2);
        root_ = new node_t(M_, 0);
    }
    RTree2(int M) : M_(M), m_(M / 2), depth_(0), split_(M, M / 2) { root_ = new node_t(M_, 0); }
    RTree2() : RTree2(8, 4) { }

  // ---------------------- implement deep copy, deep assignment, move constructor and move assignment
  
    // modifiers
    void insert(const SpatialObject& obj) {
        // select a leaf where insert obj
        node_t *l = choose_leaf_(obj);
        l->insert(item_t(obj));
	// propagate tree rebalancing up
	std::vector<bool> ctx(depth_, false);
	adjust_tree_(l, ctx);
	return;
    }
  
    void erase(const SpatialObject& obj) {
        // find leaf containing obj
        const auto& [l, i] = find_leaf_(obj);
        if (!l) return;
        l->erase(*i);
        condense_tree_(l);
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
    // observers
    node_t* root() { return root_; }
    int depth() const { return depth_; }

    // geometric queries
    // find all spatial objects whose bounding box intersects the given query range
    std::vector<const SpatialObject*> intersect_search(const std::array<double, 2 * embed_dim>& query) const {
        std::vector<const SpatialObject*> result;
        bbox_t query_bbox(query);
        node_t* curr;
        std::stack<node_t*> stack_;
        stack_.push(root_);
        while (!stack_.empty()) {
            curr = stack_.top();
            stack_.pop();
	    // perform intersection test
            if (!curr->is_leaf()) {
                for (const auto& item : *curr) {
                    if (query_bbox.intersects(item.bbox())) { stack_.push(item.node()); }
                }
            } else {
                for (const auto& item : *curr) {
                    if (query_bbox.intersects(item.data().bbox())) { result.push_back(std::addressof(item.data())); }
                }
            }
        }
        return result;
    }

    // dfs memory deallocation
    ~RTree2() {
        std::stack<node_t*> stack_;
        stack_.push(root_);
        node_t* curr;
        while (!stack_.empty()) {
            curr = stack_.top();
            stack_.pop();
            if (!curr->is_leaf()) {
                for (const auto& item : *curr) { stack_.push(item.node()); }
            }
            delete curr;
        }
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_R_TREE_H__
