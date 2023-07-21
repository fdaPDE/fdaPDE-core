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
#include <unordered_map>

namespace fdapde {
namespace core {

// An implementation of a binary tree data structure

// forward declaration
template <typename T> class BinaryNode;
template <typename T> class BinaryTree;
enum LinkDirection { LEFT = 0, RIGHT = 1 };

// binary node for type T
template <typename T> class BinaryNode {
   private:
    typedef BinaryNode<T> NodeType;
    T data_;
    std::array<NodeType*, 2> nodes_;   // left and right childs
    NodeType* father_;                 // pointer to father
    std::size_t node_id_;              // node identifier

    NodeType* insert(const T& node_data, std::size_t ID, int link_direction) {
        if (nodes_[link_direction] != nullptr) return nullptr;   // no possible insertion
        NodeType* new_node = new BinaryNode<T>(node_data, ID, this);
        // attach node and return
        nodes_[link_direction] = new_node;
        return new_node;
    }
    void remove(int link_direction) {
        delete nodes_[link_direction];      // free memory
        nodes_[link_direction] = nullptr;   // point dangling pointer to null
    }
   public:
    bool visited = false;   // can be used to flag this node as visited in a visit algorithm

    // constructors
    BinaryNode() = default;
    BinaryNode(const T& data, std::size_t node_id, BinaryNode<T>* father) :
        data_(data), node_id_(node_id), father_(father), nodes_({nullptr, nullptr}) {};

    // insertion
    NodeType* insert_left(const T& node_data, std::size_t id)  { return insert(node_data, id, LinkDirection::LEFT);  }
    NodeType* insert_right(const T& node_data, std::size_t id) { return insert(node_data, id, LinkDirection::RIGHT); }
    NodeType* insert(const T& node_data, std::size_t id) {
        return insert_left(node_data, id) ? nodes_[LinkDirection::LEFT] : insert_right(node_data, id);
    }
    // deletion
    void remove_left()  { remove(LinkDirection::LEFT ); }
    void remove_right() { remove(LinkDirection::RIGHT); }

    // getters
    bool is_leaf() const { return !nodes_[0] && !nodes_[1]; }   // check if node has no children attached
    const T& data() const { return data_; }
    NodeType* father() { return father_; }
    std::size_t ID() const { return node_id_; }
    NodeType* left_node()  { return nodes_[LinkDirection::LEFT ]; }
    NodeType* right_node() { return nodes_[LinkDirection::RIGHT]; }

    // BinaryTree destructor will free memory
    ~BinaryNode() = default;
};

// Binary tree implementation for nodes of type T.
template <typename T> class BinaryTree {
   private:
    typedef BinaryNode<T> NodeType;
    typedef std::size_t Index;

    NodeType* root_ = nullptr;
    Index n_nodes_ = 0;
    Index max_id_ = 0;
    std::unordered_map<Index, NodeType*> node_map_;   // O(1) access to node given its ID

    // deep copy
    BinaryTree<T>& copy_(const BinaryTree<T>& other) {
        if (root_ != nullptr) dealloc();                                           // dealloc this tree
        root_ = new NodeType(other.root()->data(), other.root()->ID(), nullptr);   // set new root
        node_map_[root_->ID()] = root_;
        n_nodes_ = other.n_nodes_;
        max_id_ = other.max_id_;

        NodeType* other_node = other.root();
        NodeType* node = root_;
        // start DFS visit
        while (other_node) {
            NodeType* left_node = other_node->left_node();
            if (left_node && !node->left_node()) {
                node_map_[left_node->ID()] = node->insert_left(left_node->data(), left_node->ID());
            } else {
                NodeType* right_node = other_node->right_node();
                if (right_node && !node->right_node()) {
                    node_map_[right_node->ID()] = node->insert_right(right_node->data(), right_node->ID());
                } else {
                    other_node = other_node->father();
                    node = node->father();
                }
            }
        }
        return *this;
    }
   public:
    // constructor
    BinaryTree() = default;
    BinaryTree(const T& root_data, Index root_id) :
        max_id_(root_id), n_nodes_(1) {   // initialize tree with a non 0 ID for the root node
        root_ = new NodeType(root_data, root_id, nullptr);
        node_map_[root_id] = root_;
    }
    BinaryTree(const T& root_data) : BinaryTree(root_data, 0) { }

    // copy assignment and constructor
    BinaryTree<T>& operator=(const BinaryTree<T>& other) { return copy_(other); }
    BinaryTree(const BinaryTree<T>& other) { copy_(other); }

    // insert node at first available position with given ID
    bool insert(const T& data, Index ID) {
        // perform a level-order traversal to find first available position
        std::queue<NodeType*> queue;
        queue.push(root());
        while (!queue.empty()) {
            // get pointer to node
            NodeType* node = queue.front();
            queue.pop();
            NodeType* inserted = node->insert(data, ID, node);
            if (inserted) {
                node_map_[ID] = inserted;
                n_nodes_++;
                max_id_ = (ID > max_id_ ? ID : max_id_);
                return true;
            } else {
                queue.push(node->left_node());
                queue.push(node->right_node());
            }
        }
        return false;
    }
    // insert node at first available position using the next available ID
    bool insert(const T& data) { return insert(data, max_id_ + 1); };
    // insert as child of father with given ID and given direction
    bool insert(const T& data, Index ID, Index father, LinkDirection direction) {
        NodeType* node = node_map_.at(father);
        NodeType* inserted =
          direction == LinkDirection::LEFT ? node->insert_left(data, ID) : node->insert_right(data, ID);

        if (inserted) {
            node_map_[ID] = inserted;
            n_nodes_++;
            max_id_ = (ID > max_id_ ? ID : max_id_);
            return true;
        }
        return false;
    }
    // insert as child of father with next available ID and given direction
    bool insert(const T& data, Index father, LinkDirection direction) {
        return insert(data, max_id_ + 1, father, direction);
    }

    // getters
    NodeType* node(Index ID) const { return node_map_.at(ID); }
    NodeType* root() const { return root_; }
    Index n_nodes() const { return n_nodes_; }

    // destructor
    void dealloc() {
        for (auto it = node_map_.begin(); it != node_map_.end(); ++it) { delete it->second; }
    }
    ~BinaryTree() { dealloc(); }
};

}   // namespace core
}   // namespace fdapde

#endif   // __BINARY_TREE_H__
