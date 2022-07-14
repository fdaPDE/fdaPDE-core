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

#include "header_check.h"

namespace fdapde {

// a nurbs basis of build over a given set of knots and weights
// forward declaration of template class, to be specialized for each value of M
// M = embedding dimension
    template<int M> class NurbsBasis {
        private:
            std::array<int,M> degree_;
            std::array<std::vector<double>,M> knots_;
            std::vector<Nurbs<M>> basis_ {};
            std::array<bool, M> periodicity_ {};

        public:
            static constexpr int StaticInputSize = M;
            //static constexpr int Order = Dynamic;
            // constructors
            constexpr NurbsBasis() : degree_({0}) { } 

            //template <typename KnotsVectorType> da capire come fare
            //   requires(requires(KnotsVectorType knots, int i) {
            //               { knots[i] } -> std::convertible_to<std::vector<double>>;
            //               { knots.size() } -> std::convertible_to<std::size_t>;
            //          })
            NurbsBasis(const std::array<std::vector<double>,M>& knots,MdArray<double, 
                full_dynamic_extent_t<M>> weights, std::array<int,M> degree,
                std::array<bool, M> periodicity = {}) : degree_(degree), periodicity_(periodicity) {
                // define basis system
                for(int i=0;i<M;i++){
                    int n = knots[i].size();
                    knots_[i].resize(n + 2 * degree_[i]);
                    knots_[i] = pad_knots(knots[i], degree_[i]);
                }
                int basis_size=1;
                for(std::size_t i=0; i< M;++i){
                    basis_size*=(knots_[i].size()-degree_[i]-1); // tensor product dim = product of dims
                }
                //basis_.reserve(basis_size);
                

                // loop over all the possible combinations of the knots, full with zeros
                std::array<int, M> index = {0};

                // instantialize the shared pointers of spline basis functions for each dimension
                std::array<std::shared_ptr<BSplineBasis>, M> M_spline_basis;
                
                for(int k=0;k<M;++k){
                    //M_spline_basis[k] = std::make_shared<BSplineBasis>(knots_[k], degree_[k]); //periodicity_[k]
                    //std::cout<<"Ecco la periodicity: "<<periodicity_[k]<<std::endl;
                    M_spline_basis[k] = std::make_shared<BSplineBasis>(knots_[k], degree_[k], periodicity_[k]);
                }

                //std::cout<<"Basis size: "<<basis_size<<std::endl;
                    
                
                for(int i=0;i<basis_size;++i){
                    //std::cout<<"Basis index1: "<<i<<std::endl;
                    basis_.emplace_back(M_spline_basis, weights, index);
                    //std::cout<<"Basis index2: "<<i<<std::endl;
                    //basis_.emplace_back(knots_, weights, index, degree);
                    // Update the index with carry-over logic
                    //std::size_t j = M - 1;
                    std::size_t j = 0;
                    // Increment the last index
                    ++index[j];
                    // Carry-over when reaching the maximum allowed size
                    while (j < M - 1 && index[j] == knots_[j].size() - degree_[j] - 1) {
                        index[j] = 0;
                        ++j;
                        ++index[j];
                    }
                //std::cout<<"Basis index: "<<std::endl;
                }
                //std::cout<<"Finito: "<<basis_.size()<<std::endl;

                    
            }
            
            // overload constructor for 1D case TO DO

            // function multiindex to index
            constexpr int multiindex_to_index(const std::array<int, M>& multiIndex) const {
                int idx = 0;
                int stride = 1;
                for (int j = 0; j < M; ++j) {
                    idx += multiIndex[j] * stride;
                    stride *= (knots_[j].size() - degree_[j] - 1);
                }
                return idx;
            }

            const Nurbs<M>& operator()(const std::array<int, M>& multiIndex) const {
                return basis_[multiindex_to_index(multiIndex)];
            }

            // given a flattened index return the multiindex
            constexpr std::array<int, M> index_to_multiindex(int idx) const {
                std::array<int, M> multiIndex;
                for (int j = 0; j < M; ++j) {
                    int dim_size = knots_[j].size() - degree_[j] - 1;
                    multiIndex[j] = idx % dim_size;
                    idx /= dim_size;
                }
                return multiIndex;
            }
            
            //NurbsBasis(const Triangulation<M, 1>& interval, MdArray<double, full_dynamic_extent_t<M>>& weights, int degree) : NurbsBasis(interval.nodes(), weights, degree) { }
            // getters
            constexpr const Nurbs<M>& operator[](int i) const { return basis_[i]; }
            constexpr int size() const { return basis_.size(); }
            constexpr std::array<int,M> degree() const { return degree_; }
            constexpr const std::vector<Nurbs<M>>& nurbs_basis() const { return basis_; }
            constexpr const std::vector<double>& knots(int i) const { return knots_[i]; }

            auto begin() const { return basis_.begin(); }
            auto end() const { return basis_.end(); }
    };
}   // namespace fdapde

#endif // __NURBS_BASIS_H__