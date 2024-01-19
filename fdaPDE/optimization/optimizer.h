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

#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include "../utils/symbols.h"
#include "../utils/type_erasure.h"

namespace fdapde {
namespace core {

// a type-erasure wrapper for an optimization algorithm optimizing objectives of type F
template <typename F> struct Optimizer__ {
    static constexpr int N = F::DomainDimension;
    using VectorType = typename std::conditional<N == Dynamic, DVector<double>, SVector<N>>::type;
    // function pointers forwardings
    template <typename O> using fn_ptrs = fdapde::mem_fn_ptrs<&O::template optimize<F>, &O::optimum, &O::value>;
    // implementation
    template <typename T> VectorType optimize(F& objective, const T& x0) {
        return fdapde::invoke<VectorType, 0>(*this, objective, x0);
    }
    VectorType optimum() { return fdapde::invoke<VectorType, 1>(*this); }
    double value() { return fdapde::invoke<double, 2>(*this); }
};
template <typename F> using Optimizer = fdapde::erase<fdapde::heap_storage, Optimizer__<F>>;

}   // namespace core
}   // namespace fdapde

#endif   // __OPTIMIZER_H__
