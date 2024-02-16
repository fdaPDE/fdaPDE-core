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

#ifndef __BINARY_TREE_H__
#define __BINARY_TREE_H__

#include <queue>
#include <stack>
#include "../concepts.h"

namespace fdapde {
namespace core {
  
template <typename T> class BinaryTree {
   public:
    using value_type = T;
    struct node_type {
        value_type data_;
        node_type *parent_ = nullptr, *l_child_ = nullptr, *r_child_ = nullptr;
        int depth_ = 0;
        // constructors
        node_type() = default;
        node_type(const T& data, int depth) : data_(data), depth_(depth) { }
        node_type(T&& data, int depth) : data_(data), depth_(depth) { }
    };
    using node_pointer = node_type*;

    BinaryTree() = default;
    // construct with root set to root_data
  explicit BinaryTree(const T& root_data) : n_nodes_(1) { root_ = new node_type(root_data, 0); }
    // copy semantic
    BinaryTree(const BinaryTree& other) { clone_(other); }
    BinaryTree<T>& operator=(const BinaryTree& other) {
        clone_(other);
        return *this;
    }
    // move semantic
    BinaryTree(BinaryTree&& other) { move_(std::move(other)); }
    BinaryTree<T>& operator=(BinaryTree&& other) {
        move_(std::move(other));
        return *this;
    }

    // iterator support
    template <typename iter> struct iterator_base {
       protected:
        node_pointer node = nullptr;
       public:
        iterator_base() = default;
        iterator_base(node_pointer node_) : node(node_) { }
        bool operator==(const iter& other) const { return other.node == node; }
        bool operator!=(const iter& other) const { return other.node != node; }
        value_type& operator*() const { return node->data_; }
        value_type* operator->() const { return &(node->data_); }
        iter l_child() const { return iter(node->l_child_); }
        iter r_child() const { return iter(node->r_child_); }
        operator bool() const { return node; }
        int depth() const { return node->depth_; }
    };

    // depth-first tree traversal iterator
    class dfs_iterator : public iterator_base<dfs_iterator> {
        std::stack<node_pointer> stack_;
       public:
        using iterator_base<dfs_iterator>::node;
        dfs_iterator() = default;
        dfs_iterator(node_pointer node) : iterator_base<dfs_iterator>(node) { stack_.push(node); }
        dfs_iterator& operator++() {
            stack_.pop();   // remove currently visited node
            if (node->r_child_) stack_.push(node->r_child_);
            if (node->l_child_) stack_.push(node->l_child_);
            if (stack_.empty()) {   // end of iterator
                node = nullptr;
                return *this;
            }
            node = stack_.top();
            return *this;
        }
    };
    dfs_iterator dfs_begin() const { return dfs_iterator(root_); }
    dfs_iterator dfs_end() const { return dfs_iterator(nullptr); }
    // start depth first traversal starting from pos
    template <typename iter> dfs_iterator dfs_begin(const iter& pos) const { return dfs_iterator(pos.node); }

    // breadth-first tree traversal iterator
    class bfs_iterator : public iterator_base<bfs_iterator> {
        std::queue<node_pointer> queue_;
       public:
        using iterator_base<bfs_iterator>::node;
        bfs_iterator() = default;
        bfs_iterator(node_pointer node) : iterator_base<bfs_iterator>(node) { queue_.push(node); }
        bfs_iterator& operator++() {
            queue_.pop();   // remove currently visited node
            if (node->l_child_) queue_.push(node->l_child_);
            if (node->r_child_) queue_.push(node->r_child_);
            if (queue_.empty()) {   // end of iterator
                node = nullptr;
                return *this;
            }
            node = queue_.front();
            return *this;
        }
    };
    bfs_iterator bfs_begin() const { return bfs_iterator(root_); }
    bfs_iterator bfs_end() const { return bfs_iterator(nullptr); }
    // start depth first traversal starting from pos
    template <typename iter> bfs_iterator bfs_begin(const iter& pos) const { return bfs_iterator(pos.node); }

    // iterates over all leafs of the tree rooted at given node, using a dfs strategy (O(n) complexity)
    class leaf_iterator : public iterator_base<leaf_iterator> {
        std::stack<node_pointer> stack_;
       public:
        using iterator_base<leaf_iterator>::node;
        leaf_iterator() = default;
        leaf_iterator(node_pointer node) : iterator_base<leaf_iterator>(node) {
            // search for first leaf in the subtree rooted at node
            stack_.push(node);
            if (node != nullptr) { operator++(); }
        }
        leaf_iterator& operator++() {
            do {
                stack_.pop();
                if (node->r_child_) stack_.push(node->r_child_);
                if (node->l_child_) stack_.push(node->l_child_);
                if (stack_.empty()) {   // end of iterator
                    node = nullptr;
                    return *this;
                }
                node = stack_.top();
            } while (node->l_child_ != nullptr || node->r_child_ != nullptr);
            return *this;
        }
    };
    leaf_iterator leaf_begin() const { return leaf_iterator(root_); }
    leaf_iterator leaf_end() const { return leaf_iterator(nullptr); }
    // explore all leaf nodes in the subtree rooted at pos
    template <typename iter> leaf_iterator leaf_begin(const iter& pos) const { return leaf_iterator(pos.node); }

    // depth-first traversal range-for loop support
    dfs_iterator begin() { return dfs_iterator(root_); }
    dfs_iterator end() { return dfs_iterator(nullptr); }
    dfs_iterator cbegin() const { return dfs_iterator(root_); }
    dfs_iterator cend() const { return dfs_iterator(nullptr); }

    // getters
    dfs_iterator root() const { return dfs_iterator(root_); }
    template <typename iter> bool is_leaf(const iter& pos) {
        return pos.node->l_child_ == nullptr && pos.node->r_child_ == nullptr;
    }
    bool empty() const { return n_nodes_ == 0; }
    int size() const { return n_nodes_; }

    // push with custom compare strategy
    template <typename Compare_>
        requires fdapde::LessThanComparable<Compare_, T, node_pointer>
    dfs_iterator push(const T& data, Compare_&& less_than) {
        if (!root_) {   // if tree empty, insert root and return
            root_ = new node_type(data, 0);
            n_nodes_++;
            return dfs_iterator(root_);
        }
	int depth = 1;
        node_pointer current = root_;
        node_pointer* tmp;
        while (true) {   // cycle until insertion point not found
            tmp = less_than(data, current) ? &current->l_child_ : &current->r_child_;
            if (!(*tmp)) {
                *tmp = new node_type(data, depth);
                (*tmp)->parent_ = current;
                n_nodes_++;
                return dfs_iterator(*tmp);
            }
            current = *tmp;
	    depth++;
        }
    }
    template <typename Compare_>
        requires fdapde::LessThanComparable<Compare_, T, node_pointer>
    dfs_iterator push(const std::initializer_list<T>& data, Compare_&& less_than) {
        auto it = data.begin();
        for (; it != data.end() - 1; ++it) push(*it, std::forward<Compare_>(less_than));
        return push(*it, std::forward<Compare_>(less_than));
    }

    // insert node as left/right child of node pointed by pos
    template <typename iter> iter push_left (const iter& pos, const T& data) { return push_(pos, true,  data); }
    template <typename iter> iter push_right(const iter& pos, const T& data) { return push_(pos, false, data); }

    // emplace node as child of node pointed by pos, replaces stored valud if child already present
    template <typename iter, typename... Args> iter emplace(const iter& pos, bool to_left, Args&&... args) {
        node_pointer* tmp = to_left ? &(pos.node->l_child_) : &(pos.node->r_child_);
        if (*tmp) {   // node already present, replace stored value
            (*tmp)->data_ = value_type(std::forward<Args>(args)...);
        } else {
            node_pointer new_node = new node_type(std::forward<Args>(args)..., (pos.node->depth_ + 1));
            new_node->parent_ = pos.node;
            *tmp = new_node;
            n_nodes_++;
        }
        return iter(*tmp);
    }
    // emplace node as left/right child of node pointer by pos
    template <typename iter, typename... Args> iter emplace_left(const iter& pos, Args&&... args) {
        return emplace(pos, true, std::forward<Args>(args)...);
    }
    template <typename iter, typename... Args> iter emplace_right(const iter& pos, Args&&... args) {
        return emplace(pos, false, std::forward<Args>(args)...);
    }

    // replaces element pointed by pos with elem, deal
    // template <typename iter> iter erase(const iter& pos) { }

    // removes all nodes rooted at node pointed by pos
    template <typename iter> void erase(iter pos) {
        node_pointer tmp;
        for (auto it = dfs_begin(pos); it != dfs_end(); ++it, delete tmp) { tmp = it.node; }
        root_ = nullptr;
        n_nodes_ = 0;
    }
    // removes all nodes of the tree
    void clear() {
        if (!empty()) erase(dfs_iterator(root_));
    }

    // read/write access to data stored by node pointer by pos
    template <typename iter> T& operator[](const iter& pos) { return pos.node->data_; }
    template <typename iter> const T& at(const iter& pos) const { return pos.node->data_; }

    ~BinaryTree() { clear(); }
   private:
    // insert node as child of node pointed by pos, replaces stored valud with data if child already present
    template <typename iter> iter push_(const iter& pos, bool to_left, const T& data) {
        node_pointer* tmp = to_left ? &(pos.node->l_child_) : &(pos.node->r_child_);
        if (*tmp) {   // node already present, replace stored value
            (*tmp)->data_ = data;
        } else {
            node_pointer new_node = new node_type(data, (pos.node->depth_ + 1));
            new_node->parent_ = pos.node;
            *tmp = new_node;
            n_nodes_++;
        }
        return iter(*tmp);
    }
    // sets this tree as a deep copy of other
    void clone_(const BinaryTree& other) {
        fdapde_assert(!other.empty());
        clear();
        // insert new root and clone by bfs visit
        root_ = new node_type(*other.root(), 0);
        n_nodes_++;
        std::queue<bfs_iterator> queue_;
        queue_.push(root_);
        bfs_iterator current = queue_.front();
        for (auto it = other.bfs_begin(); it != other.bfs_end(); ++it) {
            queue_.pop();
            if (it.node->l_child_) queue_.push(push_left(current, it.node->l_child_->data_));
            if (it.node->r_child_) queue_.push(push_right(current, it.node->r_child_->data_));
            if (!queue_.empty()) current = queue_.front();
        }
    }
    // move resources from other to this tree by just setting root's pointers
    void move_(BinaryTree&& other) {
        clear();
        if (other.empty()) return;
        // set new root and child pointers to other
        root_ = new node_type(other.root_->data_, 0);
        root_->l_child_ = other.root_->l_child_;
        root_->r_child_ = other.root_->r_child_;
        delete other.root_;   // free other's root memory
        other.root_ = nullptr;
        n_nodes_ = other.n_nodes_;
        other.n_nodes_ = 0;
    }

    node_pointer root_ = nullptr;
    int n_nodes_ = 0;
};

// Binary Search Tree implementation, adaptor of the BinaryTree type
template <typename T> class BST {
   private:
    using Container_ = BinaryTree<T>;
    struct Compare {
        bool operator()(const T& data, const Container_::node_pointer& node) const { return data < node->data_; }
    };
    Container_ tree_;
   public:
    using value_type    = Container_::value_type;
    using node_type     = Container_::node_type;
    using node_pointer  = Container_::node_pointer;
    using dfs_iterator  = Container_::dfs_iterator;
    using bfs_iterator  = Container_::bfs_iterator;
    using leaf_iterator = Container_::leaf_iterator;
  
    // constructors
    BST() = default;
    explicit BST(const T& root_data) : tree_(root_data) {};
    BST(const BST& other) : tree_(other) {};
    BST<T>& operator=(const BST& other) {
        tree_(other);
        return *this;
    }
    BST(BST&& other) : tree_(std::move(other)) {};
    BST<T>& operator=(BST&& other) {
        tree_(std::move(other));
        return *this;
    }

    // depth-first traversal iterators
    dfs_iterator dfs_begin() const { return tree_.dfs_begin(); }
    dfs_iterator dfs_begin(const dfs_iterator& pos) const { return tree_.dfs_begin(pos); }
    dfs_iterator dfs_end() const { return tree_.dfs_end(); }
    // breadth-first traversal iterators
    bfs_iterator bfs_begin() const { return tree_.bfs_begin(); }
    bfs_iterator bfs_begin(const bfs_iterator& pos) const { return tree_.bfs_begin(pos); }
    bfs_iterator bfs_end() const { return tree_.bfs_end(); }
    // leaf traversal iterators
    leaf_iterator leaf_begin() const { return tree_.leaf_begin(); }
    leaf_iterator leaf_begin(const leaf_iterator& pos) const { return tree_.leaf_begin(pos); }
    leaf_iterator leaf_end() const { return tree_.leaf_end(); }
    // range-for loop by depth-first traversal
    dfs_iterator begin() { return tree_.begin(); }
    dfs_iterator end() { return tree_.end(); }
    dfs_iterator cbegin() const { return tree_.cbegin(); }
    dfs_iterator cend() const { return tree_.cend(); }
    // insertion
    dfs_iterator push(const T& data) { return tree_.push(data, Compare {}); }
    dfs_iterator push(const std::initializer_list<T>& data) { return tree_.push(data, Compare {}); }
    BST& operator=(const std::initializer_list<T>& data) {
        tree_.clear();
        push(data);
        return *this;
    }
    // deletion
    template <typename iter> void erase(iter pos) { tree_.erase(pos); }
    void clear() { tree_.clear(); }
    // utilities
    template <typename iter> T& operator[](const iter& pos) { return tree_[pos]; }
    template <typename iter> const T& at(const iter& pos) const { return tree_.at(pos); }
    dfs_iterator root() const { return tree_.root(); }
    bool empty() const { return tree_.empty(); }
    int size() const { return tree_.size(); }

    // search in O(log(n)) time, return end iterator if element is not in the tree
    dfs_iterator find(const T& data) const {
        node_pointer current = tree_.root().node;
        node_pointer* tmp;
        while (current->l_child_ || current->r_child_) {   // cycle until leaf not found
            if (current->data_ == data) { return dfs_iterator(current); }
            // move down
            current = Compare {}(data, current) ? current->l_child_ : current->r_child_;
        }
        return tree_.cend();
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __BINARY_TREE_H__
