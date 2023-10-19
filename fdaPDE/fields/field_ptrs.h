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

#ifndef __FIELD_PTRS_H__
#define __FIELD_PTRS_H__

#include "matrix_expressions.h"
#include "scalar_expressions.h"
#include "vector_expressions.h"

namespace fdapde {
namespace core {

// basic pointer type for scalar expressions
template <typename E> class ScalarPtr : public ScalarExpr<E::static_inner_size, ScalarPtr<E>> {
    static_assert(std::is_base_of<ScalarBase, E>::value);
   private:
    typename std::remove_reference<E>::type* ptr_;
   public:
    // constructor
    ScalarPtr(E* ptr) : ptr_(ptr) {};
    // delegate to pointed memory location
    double operator()(const SVector<E::static_inner_size>& p) const { return ptr_->operator()(p); }
    template <typename T> void forward(T i) { ptr_->forward(i); }
    // access to pointed element
    E* operator->() { return ptr_; }
    typedef E PtrType;   // expose wrapped type
};

// basic pointer type for vectorial expressions
template <typename E> class VectorPtr : public VectorExpr<E::static_inner_size, E::rows, VectorPtr<E>> {
    static_assert(std::is_base_of<VectorBase, E>::value);
   private:
    typename std::remove_reference<E>::type* ptr_;
   public:
    VectorPtr(E* ptr) : ptr_(ptr) {};
    // delegate to pointed memory location
    auto operator[](std::size_t i) const { return ptr_->operator[](i); }
    template <typename T> void forward(T i) { ptr_->forward(i); }
    // access to pointed element
    E* operator->() { return ptr_; }
    typedef E PtrType;   // expose wrapped type
};

// basic pointer type for matrix expressions
template <typename E> class MatrixPtr : public MatrixExpr<E::static_inner_size, E::rows, E::cols, MatrixPtr<E>> {
    static_assert(std::is_base_of<MatrixBase, E>::value);
   private:
    typename std::remove_reference<E>::type* ptr_;
   public:
    MatrixPtr(E* ptr) : ptr_(ptr) {};
    // delegate to pointed memory location
    auto coeff(std::size_t i, std::size_t j) const { return ptr_->coeff(i, j); }
    template <typename T> void forward(T i) { ptr_->forward(i); }
    // access to pointed element
    E* operator->() { return ptr_; }
    typedef E PtrType;   // expose wrapped type
};

}   // namespace core
}   // namespace fdapde

#endif   // __FIELD_PTRS_H__
