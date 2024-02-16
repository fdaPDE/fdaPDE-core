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

#include <fdaPDE/utils.h>
#include <gtest/gtest.h>   // testing framework

using fdapde::core::BinaryTree;
using fdapde::core::BST;

TEST(binary_tree_test, bfs_dfs_visit_iterators) {
    BinaryTree<int> tree(1);
    auto n1 = tree.push_left(tree.root(), 2);
    auto n2 = tree.push_right(tree.root(), 7);
    auto n3 = tree.push_left(n1, 3);
    tree.push_right(n1, 4);
    tree.push_left(n3, 5);
    tree.push_right(n3, 6);
    auto n4 = tree.push_left(n2, 8);
    tree.emplace_left(n4, 9);
    
    // assert depth first traversal is correct
    std::vector<int> expected_dfs_order = {1, 2, 3, 5, 6, 4, 7, 8, 9};
    std::size_t i = 0;
    for (auto it = tree.dfs_begin(); it != tree.dfs_end(); ++it, ++i) { EXPECT_TRUE(*it == expected_dfs_order[i]); }
    // assert bredth first traversal is correct
    i = 0;
    std::vector<int> expected_bfs_order = {1, 2, 7, 3, 4, 8, 5, 6, 9};
    for (auto it = tree.bfs_begin(); it != tree.bfs_end(); ++it, ++i) { EXPECT_TRUE(*it == expected_bfs_order[i]); }
    // non const access by depth first traversal
    i = 0;
    for (auto& node_value : tree) { node_value = node_value * 2; }
    for (const auto& node_value : tree) {
        EXPECT_TRUE(node_value == 2 * expected_dfs_order[i]);
        ++i;
    }
    // DFS visit of subtree rooted at 2
    i = 0;
    std::vector<int> expected_dfs_subtree = {2, 3, 5, 6, 4};
    for (auto it = tree.dfs_begin(n1); it != tree.dfs_end(); ++it, ++i) {
        EXPECT_TRUE(*it == 2 * expected_dfs_subtree[i]);
    }
    // change value pointed by n3, check whole subtree rooted at 2
    tree[n3] = 17;   // subscript operator
    i = 0;
    expected_dfs_subtree = {4, 17, 10, 12, 8};
    for (auto it = tree.dfs_begin(n1); it != tree.dfs_end(); ++it, ++i) {
        EXPECT_TRUE(*it == expected_dfs_subtree[i]);
    }    
}

TEST(binary_tree_test, leaf_iterator) {
    BinaryTree<int> tree(1);
    auto n1 = tree.push_left(tree.root(), 2);
    auto n2 = tree.push_right(tree.root(), 7);
    auto n3 = tree.push_left(n1, 3);
    tree.push_right(n1, 4);
    tree.push_left(n3, 5);
    tree.push_right(n3, 6);
    auto n4 = tree.push_left(n2, 8);
    tree.push_left(n4, 9);

    // map of expected leafs in tree
    std::unordered_map<int, bool> is_leaf_map {
      {1, false},
      {2, false},
      {3, false},
      {4, true },
      {5, true },
      {6, true },
      {7, false},
      {8, false},
      {9, true }
    };
    // check all asserted leafs are actually so
    for (auto it = tree.dfs_begin(); it != tree.dfs_end(); ++it) {
        EXPECT_TRUE(tree.is_leaf(it) == is_leaf_map.at(*it));
    }
    // using leaf iterator
    for (auto it = tree.leaf_begin(); it != tree.leaf_end(); ++it) {
        EXPECT_TRUE(is_leaf_map.at(*it));
    }
}

TEST(binary_tree_test, copy_constructor_assignment) {
    BinaryTree<int> tree(1);
    auto n1 = tree.push_left(tree.root(), 2);
    auto n2 = tree.push_right(tree.root(), 7);
    auto n3 = tree.push_left(n1, 3);
    tree.push_right(n1, 4);
    tree.push_left(n3, 5);
    tree.push_right(n3, 6);
    auto n4 = tree.push_left(n2, 8);
    tree.push_left(n4, 9);

    // copy construct another tree
    BinaryTree<int> copy(tree);
    EXPECT_TRUE(copy.size() == tree.size());
    // assert depth first traversal is correct
    std::vector<int> expected_dfs_order = {1, 2, 3, 5, 6, 4, 7, 8, 9};
    std::size_t i = 0;
    for (auto it = copy.dfs_begin(); it != copy.dfs_end(); ++it, ++i) { EXPECT_TRUE(*it == expected_dfs_order[i]); }

    // assign another tree
    BinaryTree<int> assigned = tree;
    EXPECT_TRUE(assigned.size() == tree.size());
    i = 0;
    for (auto it = assigned.dfs_begin(); it != assigned.dfs_end(); ++it, ++i) {
        EXPECT_TRUE(*it == expected_dfs_order[i]);
    }
}

TEST(binary_tree_test, move_constructor_assignment) {
    BinaryTree<int> tree(1);
    auto n1 = tree.push_left(tree.root(), 2);
    auto n2 = tree.push_right(tree.root(), 7);
    auto n3 = tree.push_left(n1, 3);
    tree.push_right(n1, 4);
    tree.push_left(n3, 5);
    tree.push_right(n3, 6);
    auto n4 = tree.push_left(n2, 8);
    tree.push_left(n4, 9);

    // move construct another tree
    BinaryTree<int> moved(std::move(tree));
    EXPECT_TRUE(moved.size() == 9);
    EXPECT_TRUE(tree.size() == 0);
    // assert depth first traversal is correct in moved tree
    std::vector<int> expected_dfs_order = {1, 2, 3, 5, 6, 4, 7, 8, 9};
    std::size_t i = 0;
    for (auto it = moved.dfs_begin(); it != moved.dfs_end(); ++it, ++i) { EXPECT_TRUE(*it == expected_dfs_order[i]); }
    // assert original tree has no nodes
    i = 0;
    for (auto it = tree.dfs_begin(); it != tree.dfs_end(); ++it, ++i);
    EXPECT_TRUE(i == 0);

    // move assign
    BinaryTree<int> assigned = std::move(moved);
    EXPECT_TRUE(assigned.size() == 9);
    EXPECT_TRUE(moved.size() == 0);
    i = 0;
    for (auto it = assigned.dfs_begin(); it != assigned.dfs_end(); ++it, ++i) {
        EXPECT_TRUE(*it == expected_dfs_order[i]);
    }
    // assert original tree has no nodes
    i = 0;
    for (auto it = moved.dfs_begin(); it != moved.dfs_end(); ++it, ++i);
    EXPECT_TRUE(i == 0);
}

TEST(binary_tree_test, bst_sorted_insertion) {
    BST<int> tree(7);
    tree.push({2, 1, 3, 4, 5, 6, 8, 9});   // use brace-initializer list

    // assert depth first traversal is correct
    std::vector<int> expected_dfs_order = {7, 2, 1, 3, 4, 5, 6, 8, 9};
    std::size_t i = 0;
    for (auto it = tree.dfs_begin(); it != tree.dfs_end(); ++it, ++i) { EXPECT_TRUE(*it == expected_dfs_order[i]); }
    EXPECT_TRUE(tree.size() == 9);

    // braced-list initialization
    tree.clear();
    EXPECT_TRUE(tree.size() == 0);
    tree = {7, 2, 1, 3, 4, 5, 6, 8};
    tree.push(9);
    i = 0;
    for (auto it = tree.dfs_begin(); it != tree.dfs_end(); ++it, ++i) { EXPECT_TRUE(*it == expected_dfs_order[i]); }
    EXPECT_TRUE(tree.size() == 9);    
}

TEST(binary_tree_test, bst_search) {
    BST<int> tree(7);
    tree.push({2, 1, 3, 4, 5, 6, 8, 9});   // use brace-initializer list

    auto it = tree.find(5);
    EXPECT_TRUE(it != tree.end() && *it == 5);
    EXPECT_TRUE(it.depth() == 4);
    // try to find an element which is not contained
    EXPECT_TRUE(tree.find(19) == tree.end());
}
