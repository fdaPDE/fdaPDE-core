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

#ifndef __KD_TREE_H__
#define __KD_TREE_H__

#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "../utils/symbols.h"
#include "../utils/data_structures/binary_tree.h"

namespace fdapde {
namespace core {

// KD-tree data structure to solve nearest neighbors search problems (uses a median splitting plane strategy)
template <int K> class KDTree {
   private:
    using Container    = BinaryTree<int>;
    using node_type    = Container::node_type;
    using node_pointer = Container::node_pointer;
    using iterator     = Container::dfs_iterator;
    using data_type = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    Container kdtree_;   // the actual BinaryTree container
    data_type data_;     // set of data indexed by this tree
   public:
    // computes the kd-tree structure for a set of points
    KDTree() = default;
    template <typename DataType_> explicit KDTree(DataType_&& data) : data_(std::forward<DataType_>(data)) {
        fdapde_assert(data_.cols() == K);
        std::vector<int> ids(data_.rows());   // vector of points ids (filled from 0 up to n-1 by std::iota)
        std::iota(ids.begin(), ids.end(), 0);
        int split_dim = 0;   // current hyperplane splitting direction
        struct point_set_t {
            point_set_t(int begin_, int end_, int split_dim_) :
                begin(begin_),              // the first point in the set
                end(end_),                  // the last point in the set
                split_dim(split_dim_) { }   // the hyperplane's direction we need to split this set to
            int begin, end, split_dim;
        };
        std::stack<point_set_t> stack;
	stack.emplace(0, ids.size(), split_dim);
	// kd-tree construction
        while (!stack.empty()) {
            // find median along the split_dim dimension
            point_set_t point_set = stack.top();
            stack.pop();
            int median = point_set.begin + (point_set.end - point_set.begin) / 2;
            auto it = ids.begin();
            std::nth_element(   // O(n) median finding algorithm
              it + point_set.begin, it + median, it + point_set.end,
              [&, &dim_ = point_set.split_dim](int p1, int p2) -> bool { return data_(p1, dim_) < data_(p2, dim_); });
            // median insertion in tree structure
            kdtree_.push(*(it + median), [&](int point, const node_pointer& node) -> bool {   // O(log(n))
                return data_(point, node->depth_ % K) < data_(node->data_, node->depth_ % K);
            });
	    // prepare for next insertion
            split_dim = (split_dim + 1) % K;
            if (median - point_set.begin > 0)     stack.emplace(point_set.begin, median, split_dim);
            if (point_set.end - (median + 1) > 0) stack.emplace(median + 1, point_set.end, split_dim);
        }
    }
    // range for over node ids, you can recover the node coordinates by for(auto p : kdtree) { data_.row(p); }
    iterator begin() { return kdtree_.begin(); }
    iterator end() { return kdtree_.end(); }

    // returns an iterator to the nearest neighbor of p. Average O(log(n)) complexity (worst case is O(n))
    iterator nn_search(const DVector<double>& p) const {
        fdapde_assert(p.size() == K);
        if (kdtree_.empty()) return kdtree_.cend();   // nothing to search
        const data_type& data = data_;
        std::stack<iterator> stack;
        // given a point p \in R^K, traverse the tree until a leaf is not found, following the kd-tree structure
        auto walk_down = [&stack, &data, &p](const iterator& start) {
            if (!start) return;
            iterator it = start;
            while (it.l_child() || it.r_child()) {   // cycle until leaf not reached
                stack.push(it);
                it = p[it.depth() % K] < data(*it, it.depth() % K) ? it.l_child() : it.r_child();
            }
            stack.push(it);   // push leaf node
        };
        // initialization
        walk_down(kdtree_.root());   // intialize stack
        iterator curr, best;
        double curr_dist = 0, best_dist = std::numeric_limits<double>::infinity();
        // unwind the recursion searching for optimal nearest point
        while (!stack.empty()) {
            curr = stack.top();
            stack.pop();
            curr_dist = (data.row(*curr).transpose() - p).squaredNorm();
            if (curr_dist < best_dist) {   // update optimal point
                best = curr;
                best_dist = curr_dist;
                if (best_dist == 0) return curr;
            }
            // check if the hypersphere centered in p intersects the current splitting hyperplane
            double r = data(*curr, curr.depth() % K) - p[curr.depth() % K];
            if (r * r < best_dist) {   // hyperplane intersect, search on the other side of the splitting plane
                walk_down(r > 0 ? curr.l_child() : curr.r_child());
            }
        }
        return best;
    }

    // solves a (rectangular) range query in a K-dimensional euclidean space
    struct RangeType {
        SVector<K> ll, ur;   // lower-left and upper-right corner
    };
    // returns a set of iterators to the nodes contained in the query
    std::unordered_set<int> range_search(const RangeType& query) const {
        std::unordered_set<int> result;
        std::stack<iterator> stack;   // auxiliary stack for tree visiting
        stack.push(kdtree_.root());
        const data_type& data = data_;

        // returns true if the node indexed by it_ lies inside the query
        auto contained = [&data, &query](const iterator& it_) -> bool {
            for (int dim = 0; dim < K; ++dim) {
                if (data(*it_, dim) > query.ur[dim] || data(*it_, dim) < query.ll[dim]) return false;
            }
            return true;
        };
        // start search
        iterator it;
        while (!stack.empty()) {
            it = stack.top();
            stack.pop();
            // add node pointed by it to solution if is contained in the query
            if (contained(it)) result.insert(*it);
            // test possible intersection of left and right child subregion with query
            if (it.l_child() ? data(*it, it.depth() % K) >= query.ll[it.depth() % K] : false) stack.push(it.l_child());
            if (it.r_child() ? data(*it, it.depth() % K) <= query.ur[it.depth() % K] : false) stack.push(it.r_child());
        }
        return result;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __KD_TREE_H__
