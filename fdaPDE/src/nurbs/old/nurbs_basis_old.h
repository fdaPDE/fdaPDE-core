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

#ifndef __NURBS_BASIS_H__
#define __NURBS_BASIS_H__

#include "../geometry/interval.h"
#include "../fields/nurbs_old.h"

namespace fdapde {

// a nurbs basis of build over a given set of knots and weights
// forward declaration of template class, to be specialized for each value of M
// M = embedding dimension
    template<int M> class NurbsBasis {
        private:
            int order_;
            std::vector<Nurbs<M>> basis_ {};

        public:
            static constexpr int StaticInputSize = M;
            static constexpr int Order = Dynamic;
            // constructors
            constexpr NurbsBasis() : order_(0) { } 

            template <typename KnotsVectorType>
                requires(requires(KnotsVectorType knots, int i) {
                            { knots[i] } -> std::convertible_to<std::vector<double>>;
                            { knots.size() } -> std::convertible_to<std::size_t>;
                        })
            NurbsBasis(KnotsVectorType&& knots, MdArray<double, full_dynamic_extent_t<M>>& weights, int order) : order_(order) {
            // construct knots vector

            std::array<std::vector<double>,M> knots_ ;
            // pad the knot vector to obtain a full basis for the whole knot span [u_0, u_n]
                for(std::size_t i=0; i < M; ++i){
                    // reserve space
                    auto n=knots[i].size();
                    knots_[i].resize(n + 2 * order_);
                    for (std::size_t j = 0; j < n + 2 * order_; ++j) {
                        if (j < order_) {
                            knots_[i][j] = knots[i][0];
                        } else {
                            if (j < n + order_) {
                                knots_[i][j] = knots[i][j - order_];
                            } else {
                                knots_[i][j] = knots[i][n - 1];
                            }
                        }
                    }
                }
                // define basis system
                int basis_size=1;
                for(std::size_t i=0; i< M;++i){
                    basis_size*=(knots_[i].size()-order_-1); // tensor product dim = product of dims
                }
                basis_.reserve(basis_size);


                // loop over all the possible combinations of the knots
                std::array<int, M> index {};

                for(int i=0;i<basis_size;++i){
                    
                    basis_.emplace_back(knots_, weights, index, order_);

                    // Update the index with carry-over logic
                    std::size_t j = M - 1;
                    // Increment the last index
                    ++index[j];
                    // Carry-over when reaching the maximum allowed size
                    while (j > 0 && index[j] == knots_[j].size() - order_ - 1) {
                        index[j] = 0;  // Reset the current index
                        --j;           // Move to the previous coordinate
                        ++index[j];    // Increment the previous coordinate
                    }
                }
            }
            //NurbsBasis(const Triangulation<M, 1>& interval, MdArray<double, full_dynamic_extent_t<M>>& weights, int order) : NurbsBasis(interval.nodes(), weights, order) { }
            // getters
            constexpr const Nurbs<M>& operator[](int i) const { return basis_[i]; }
            constexpr int size() const { return basis_.size(); }
            
    };
}   // namespace fdapde

#endif // __NURBS_BASIS_H__