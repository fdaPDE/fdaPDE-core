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

#ifndef __ADT_H__
#define __ADT_H__

#include <list>
#include <set>
#include <stack>

#include "../../utils/DataStructures/Tree.h"
using fdaPDE::core::Tree;
using fdaPDE::core::node_ptr;
using fdaPDE::core::LinkDirection;
#include "../../utils/symbols.h"
#include "../mesh.h"
#include "point_locator.h"

namespace fdapde {
namespace core {

// a specific node data structure for easy management of the ADT during element search
template <unsigned int N> struct ADTnode {
    unsigned int elementID_;   // the element ID to which this node referes to
    SVector<N> point_;         // the point stored in this node
    std::pair<SVector<N>, SVector<N>> range_;   // the range in the unit hypercube this node refers to

    ADTnode(unsigned int elementID, const SVector<N>& point, const std::pair<SVector<N>, SVector<N>>& range) :
        elementID_(elementID), point_(point), range_(range) { }
};

// An N-dimensional rectangle described by its left-lower corner and upper-right corner.
template <unsigned int N> class Query {
   private:
    std::pair<SVector<N>, SVector<N>> query_range_;   // (left_lower_corner, right_upper_corner)
   public:
    // constructor
    Query(const std::pair<SVector<N>, SVector<N>>& query_range) : query_range_(query_range) { }
    // returns true if a given point lies inside the query
    bool contains(const SVector<N>& point) const {
        for (size_t dim = 0; dim < N; ++dim) {
            // return false if point lies outside
            if (point[dim] > query_range_.second[dim] || point[dim] < query_range_.first[dim]) { return false; }
        }
        return true;
    }
    // returns true if the query intersects a given rectangle
    bool intersect(const std::pair<SVector<N>, SVector<N>>& rect) const {
        // keep track along which dimension query and rect intersects
        std::array<bool, N> bool_vector {};
        for (size_t dim = 0; dim < N; ++dim) {
            // load here once, access it fast
            double qs = query_range_.second[dim], qf = query_range_.first[dim];
            double rs = rect.second[dim], rf = rect.first[dim];

            if ((qs > rf && rs > qf) || (rs > qf && qs > rf) ||   // partially overlapping sides
                (rf < qf && qs < rs) || (qf < rf && rs < rs) ||   // query_range_ contained into rect or viceversa
                (qs == qf && qs <= rs)) {                         // degenerate case
                bool_vector[dim] = true;                          // query intersect rectangle along this dimension
            }
        }
        // query and rect intersects if and only if they intersects along each dimension
        bool result = true;
        for (bool b : bool_vector) result &= b;
        return result;
    }
};

// Alternating Digital Tree implementation for tree-based point location problems
  template <unsigned int M, unsigned int N, unsigned int R> class ADT : public PointLocator<M,N,R> {
   private:
    Tree<ADTnode<2 * N>> tree;              // tree data structure to support ADT
    const Mesh<M, N, R>& mesh_;             // domain over which build the ADT
    std::array<double, N> normalization_;   // vector of range-normalization constants

    // build the ADT structure given a set of 2N-dimensional points.
    void init(const std::vector<std::pair<SVector<2 * N>, unsigned int>>& data) {
        // initialization (ll : left_lower_corner, ru : right_upper_corner)
        SVector<2 * N> ll = SVector<2 * N>::Zero(), ru = SVector<2 * N>::Ones();
        tree = Tree(ADTnode<2 * N>(data[0].second, data[0].first, std::make_pair(ll, ru)));

        // insert data in the tree
        for (size_t j = 1; j < data.size(); ++j) {
            SVector<2 * N> node_data = data[j].first;
            unsigned int node_id = data[j].second;
            std::pair<SVector<2 * N>, SVector<2 * N>> node_range = std::make_pair(ll, ru);

            node_ptr<ADTnode<2 * N>> current = tree.getNode(0);   // root node

            bool inserted = false;                 // stop iterating when an insertion point has been found
            unsigned int iteration = 1;            // split points are located at (0.5)^iteration
            std::array<double, 2 * N> offset {};   // keep track of the splits of the domain at each iteration

            // search for the right insertion location in the tree
            while (!inserted) {
                for (size_t dim = 0; dim < 2 * N; ++dim) {                         // cycle over dimensions
                    double split_point = offset[dim] + std::pow(0.5, iteration);   // split point
                    if (node_data[dim] < split_point) {
                        node_range.second[dim] = split_point;   // shrink node range on the left
                        if (tree.insert(ADTnode<2 * N>(node_id, node_data, node_range), current->getKey(),
                                        LinkDirection::LEFT) != nullptr) {   // O(1) operation
                            inserted = true;                                 // stop searching for location
                            break;
                        } else
                            current = current->getChildren()[0];   // move to left child
                    } else {
                        node_range.first[dim] = split_point;   // shrink node range on the right
                        if (tree.insert(ADTnode<2 * N>(node_id, node_data, node_range), current->getKey(),
                                        LinkDirection::RIGHT) != nullptr) {   // O(1) operation
                            inserted = true;                                  // stop searching for location
                            break;
                        } else {
                            current = current->getChildren()[1];   // move to right child
                            offset[dim] += std::pow(0.5, iteration);
                        }
                    }
                }
                // virtually perform an half split of the hyper-cube
                iteration++;
            }
        }
        return;
    }

    // performs a geometric search returning all points which lie in a given query
    std::list<std::size_t> geometric_search(const Query<2 * N>& query) const {
        std::list<std::size_t> found;
        std::stack<node_ptr<ADTnode<2 * N>>> stack;
        stack.push(tree.getNode(0));   // start from root

        while (!stack.empty()) {
            node_ptr<ADTnode<2 * N>> current = stack.top();
            stack.pop();
            // add to solution if point is contained in query range
            if (query.contains(current->getData().point_)) { found.push_back(current->getData().elementID_); }
            // get children at node
            std::array<node_ptr<ADTnode<2 * N>>, 2> children = current->getChildren();

            bool left_child_test = children[0] != nullptr ? query.intersect(children[0]->getData().range_) : false;
            bool right_child_test = children[1] != nullptr ? query.intersect(children[1]->getData().range_) : false;

            if (left_child_test)   // test if left  child range intersects query range
                stack.push(children[0]);
            if (right_child_test)   // test if right child range intersects query range
                stack.push(children[1]);
        }
        return found;
    }
   public:
    ADT(const Mesh<M, N, R>& mesh) : mesh_(mesh) {
        // move mesh elements to 2N dimensional points
        std::vector<std::pair<SVector<2 * N>, unsigned int>> data;
        data.reserve(mesh_.elements());   // avoid useless reallocations at runtime
        // computation of normalization constants
        for (std::size_t dim = 0; dim < N; ++dim) {
            normalization_[dim] = 1.0 / (mesh_.range()[dim].second - mesh_.range()[dim].first);
        }

        for (const auto& element : mesh_) {
            // compute bounding box
            std::pair<SVector<N>, SVector<N>> bounding_box = element.bounding_box();
            // create 2N dimensional point
            SVector<2 * N> element_to_point;

            // scale dimensions in the unit hypercube
            // point scaling means to apply the following linear transformation to each dimension of the point
            // scaledPoint[dim] = (point[dim] - meshRange[dim].first)/(meshRange[dim].second - meshRange[dim].first)
            for (size_t dim = 0; dim < N; ++dim) {
                bounding_box.first[dim] = (bounding_box.first[dim] - mesh_.range()[dim].first) * normalization_[dim];
                bounding_box.second[dim] = (bounding_box.second[dim] - mesh_.range()[dim].first) * normalization_[dim];
            }
            element_to_point << bounding_box.first, bounding_box.second;
            data.emplace_back(element_to_point, element.ID());
        }
        // set up internal data structure
        init(data);
    }
    // solves the point location problem
    const Element<M, N, R>* locate(const SVector<N>& p) const {
        // point scaling means to apply the following linear transformation to each dimension of the point
        // scaledPoint[dim] = (point[dim] - meshRange[dim].first)/(meshRange[dim].second - meshRange[dim].first)
        SVector<N> scaled_point;
        for (size_t dim = 0; dim < N; ++dim) {
            scaled_point[dim] = (p[dim] - mesh_.range()[dim].first) * normalization_[dim];
        }
        // build search query
        SVector<2 * N> ll, ur;
        ll << SVector<N>::Zero(), scaled_point;
        ur << scaled_point, SVector<N>::Ones();
        std::pair<SVector<2 * N>, SVector<2 * N>> query = std::make_pair(ll, ur);

        // perform search (now the problem has been transformed to the one of searching for the set
        // of points contained in the range of the query. See "(J. Bonet, J. Peraire) 1991
        // An alternating digital tree (ADT) algorithm for 3D geometric searching and intersection problems"
        // for details)
        std::list<std::size_t> found = geometric_search(Query<2 * N>(query));
        // exhaustively scan the query results to get the searched mesh element
        for (std::size_t ID : found) {
            auto element = mesh_.element(ID);
            if (element.contains(p)) { return &mesh_.element(ID); }
        }
        // no element found
        return nullptr;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __ADT_H__
