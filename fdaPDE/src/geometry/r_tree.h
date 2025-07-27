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
template <typename SpatialObject>
    requires(requires(SpatialObject obj) {
        SpatialObject::embed_dim;
        { obj.bbox() } -> std::convertible_to<std::array<double, 2 * SpatialObject::embed_dim>>;
    })
class RTree {
   private:
    static constexpr int embed_dim = SpatialObject::embed_dim;

    // internal structures
    struct bbox_t {
        bbox_t() : bbox_(), measure_(0) { std::fill(bbox_.begin(), bbox_.end(), 0); }
        bbox_t(const SpatialObject& obj) : bbox_(obj.bbox()), measure_(1.0) {
            for (int i = 0; i < embed_dim; ++i) { measure_ *= (bbox_[i + embed_dim] - bbox_[i]); }
        }
        template <typename T>
            requires(internals::is_vector_like_v<T>)
        bbox_t(const T& bbox) : bbox_(), measure_(1.0) {
            for (int i = 0; i < 2 * embed_dim; ++i) { bbox_[i] = bbox[i]; }
	    for (int i = 0; i < embed_dim; ++i) { measure_ *= (bbox_[i + embed_dim] - bbox_[i]); }
        }

        double measure() const { return measure_; }
        const std::array<double, 2 * embed_dim>& coords() const { return bbox_; }
        void expand(const bbox_t& other) {
            for (int i = 0; i < embed_dim; ++i) {
                bbox_[i] = std::min(bbox_[i], other.bbox_[i]);
                bbox_[i + embed_dim] = std::max(bbox_[i + embed_dim], other.bbox_[i + embed_dim]);
            }
        }
        template <typename PointT>
            requires(internals::is_vector_like_v<PointT>)
        bool constains(const PointT& p) const {
            fdapde_assert(p.size() == embed_dim);
            for (int i = 0; i < embed_dim; ++i) {
                if (bbox_[i] > p[i] || bbox_[i + embed_dim] < p[i]) return false;
            }
            return true;
        }
        double enlargment(const bbox_t& other) {
            // compute measure of expanded rectangle
            double exp_measure_ = 1;
            for (int i = 0; i < embed_dim; ++i) {
                exp_measure_ *=
                  (std::max(bbox_[i + embed_dim], other.bbox_[i + embed_dim]) - std::min(bbox_[i], other.bbox_[i]));
            }
            return exp_measure_ - measure_;
        }
       private:
        std::array<double, 2 * embed_dim> bbox_;   // (x1_min, x2_min, ..., xn_min, x1_max, ..., xn_max)
        double measure_;
    };

    // data item stored in node
    struct node_t;
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
        bbox_t& bbox() { return bbox_; }
        node_t* node() const {
            fdapde_assert(type_ == type_t::NODE);
            return node_;
        }
        const SpatialObject& data() const {
            fdapde_assert(type_ == type_t::DATA);
            return *data_;
        }
        // shallow destruction (not-owned memory)
        ~item_t() = default;
    };
    struct node_t {
        node_t(int M, bool is_leaf = true) : is_leaf_(is_leaf), size_(0) {
            data_.resize(M + 1);   // allow overflow to use this buffer for in-place split logic
            free_.resize(M + 1, true);
        }
        // observers
        node_t* parent() const { return parent_; }
        bool is_leaf() const { return is_leaf_; }
        bool is_root() const { return parent_ == nullptr; }
        std::size_t size() const { return size_; }
        const item_t& item(int i) const { return data_[mapped_(i)]; }
        item_t& item(int i) { return data_[mapped_(i)]; }
        bbox_t bbox() const {   // minimal bounding box containing all spatial objects rooted at this node
            if (size_ == 0) return bbox_t();
            bbox_t bbox_ = data_[mapped_(0)].bbox();
            for (int i = 1; i < size_; ++i) { bbox_.expand(data_[mapped_(i)].bbox()); }
            return bbox_;
        }
        // modifiers
        void insert(const item_t& item) {
            // find idx of first free slot in data_
            int j = 0;
            for (int i = 0, n = free_.size(); i < n; ++i) {
                if (free_[i]) {
                    j = i;
                    break;
                }
            }
            data_[j] = item;
            free_[j] = false;	    
            size_++;
        }
        void erase(int i) {
            free_[mapped_(i)] = true;
            size_--;
        }
        void set_parent(node_t* parent) { parent_ = parent; }
        void clear() {
            std::fill(free_.begin(), free_.end(), true);
	    size_ = 0;
        }

      const std::vector<bool>& free() const { return free_; }
      
       private:
        // maps i to index of i-th non-free element in data_
        int mapped_(int k) const {
            int i = 0, j = 0, n = free_.size();
            for (; i < n; ++i) {
                if (!free_[i]) {
                    if (j == k) { return i; }
                    j++;
                }
            }
	    return i;
        }
        // tree structure data
        node_t* parent_ = nullptr;
        bool is_leaf_;
        std::vector<item_t> data_;
        std::vector<bool> free_;   // free_[i] == true \iff data_[i] available for writing
        int size_ = 0;
    };

    // select a leaf node in which to place a new index entry obj
    node_t* choose_leaf_(const SpatialObject& obj) {
        bbox_t bbox(obj);
        node_t* curr = root_;
        while (!curr->is_leaf()) {
            // find the element in curr whose rectangle needs least enlargment to include obj
            double enlargment = std::numeric_limits<double>::infinity();
            int j = 0;
            for (int i = 0; i < curr->size(); ++i) {
                double tmp = bbox.enlargment(curr->item(i).bbox());
                if (tmp < enlargment) {
                    enlargment = tmp;
                    j = i;
                }
            }
            curr = curr->item(j).node();
        }
        return curr;
    }

    std::pair<node_t*, node_t*> split_node_(node_t* node) {
        // allocate memory for new node
        node_t* n = new node_t(M_);
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
                for (int i : g) {   // move data to node n
                    n->insert(node->item(i));
                    node->erase(i);
                }
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
                        b = less_than(d1, d2) ||
                                (almost_equal(d1, d2) && less_than(bbox[0].measure(), bbox[1].measure())) ?
                              0 :
                              1;
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
        for (int i : g) { node->erase(i); }
        return std::make_pair(node, n);
    }

    // ascend from a leaf node l to the root, adjusting covering rectangles and propagating node splits as necessary
    void adjust_tree_(node_t* l1, node_t* l2) {
        node_t *n1 = l1, *n2 = l2;
        while (!n1->is_root()) {
            node_t* parent = n1->parent();
            // update l's bounding box in parent
            for (int i = 0, m = parent->size(); i < m; ++i) {
                if (n1 == parent->item(i).node()) {
                    parent->item(i).bbox() = n1->bbox();
                    break;
                }
            }
            if (n2 != nullptr) {   // a split occurred
                parent->insert(item_t(n2));
		n2->set_parent(parent);
                if (parent->size() > M_) {   // split parent
                    const auto& [e1, e2] = split_node_(parent);
                    n1 = e1;
                    n2 = e2;
                    continue;
                }
            }
            n1 = parent;
            n2 = nullptr;
        }
        // if root was split, create new root
        if (n2 != nullptr) {
            node_t* new_root = new node_t(M_, false);	    
            n1->set_parent(new_root);
            n2->set_parent(new_root);
            new_root->insert(item_t(n1));
            new_root->insert(item_t(n2));
            root_ = new_root;
        }
        return;
    }

    node_t* root_ = nullptr;
    int M_;   // maximum number of entries per node
    int m_;   // minimum number of entries per node
   public:
    RTree(int M, int m) : M_(M), m_(m) {
        fdapde_assert(m_ <= M / 2);
        root_ = new node_t(M_);
    }
    RTree() : RTree(10, 5) { }

    void insert(const SpatialObject& obj) {
        // select a leaf where insert obj
        node_t* l = choose_leaf_(obj);
        node_t* ll = nullptr;   // not null only in case of split
        l->insert(item_t(obj));
        if (l->size() > M_) {   // restore consistent state (no more than M_ + 1 items per node)
            const auto& [n, nn] = split_node_(l);
	    l = n;
	    ll = nn;
        }
        adjust_tree_(l, ll);
	return;
    }
    node_t* root() { return root_; }

    // geometric queries
};

}   // namespace fdapde

#endif   // __FDAPDE_R_TREE_H__
