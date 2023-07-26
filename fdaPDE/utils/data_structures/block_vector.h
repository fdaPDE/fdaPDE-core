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

#ifndef __BLOCK_VECTOR_H__
#define __BLOCK_VECTOR_H__

namespace fdapde {
namespace core {
  
// a vector made of blocks of equal size
template <typename T> class BlockVector {
   private:
    DVector<T> data_;
    std::size_t n_;   // number of blocks
    std::size_t m_;   // size of single block
   public:
    // constructor
    BlockVector(std::size_t n, std::size_t m) : n_(n), m_(m) { data_ = DVector<T>::Zero(n * m); }
    auto operator()(std::size_t i) { return data_.block(i * m_, 0, m_, 1); }   // access to i-th block
    auto operator()(std::size_t i, std::size_t j) {
        return data_.block(i * m_, 0, j * m_, 1);
    }   // access to blocks (i, i+j)

    auto head(std::size_t i) { return data_.block(0, 0, i * m_, 1); }               // access to first i blocks
    auto tail(std::size_t i) { return data_.block((n_ - i) * m_, 0, i * m_, 1); }   // access to last i blocks
    // getter to internal data
    const DVector<T>& get() const { return data_; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __BLOCK_VECTOR_H__
