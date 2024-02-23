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

#ifndef __FDAPDE_CORE_NON_LINEARITY_BASE_H__
#define __FDAPDE_CORE_NON_LINEARITY_BASE_H__

#include <type_traits>

#include "../utils/symbols.h"
#include "scalar_expressions.h"

namespace fdapde{
namespace core{

    template <int N, typename B>
    class NonLinearityBase {
    protected:
        static constexpr std::size_t n_basis_ = B::n_basis;
        typedef std::shared_ptr<DVector<double>> VecP;

        B basis_ {};  //calls default constructor
        mutable VecP f_prev_;  // pointer to vector containing the solution on one element at the previous time step
        std::function<double(SVector<N>, SVector<1>)> h = [](SVector<N> x, SVector<1> ff) -> double {return 1 - ff[0];};

        // protected method that preforms $\sum_i {f_i*\psi_i(x)}$
        SVector<1> f(const SVector<N>& x) const{
            SVector<1> result;
            result[0] = 0;
            for (std::size_t i = 0; i < n_basis_; i++){
                result[0] += (*f_prev_)[i] * basis_[i](x); }
            return result;
        }

    public:
        // constructor
        NonLinearityBase() = default;
        NonLinearityBase(std::function<double(SVector<N>, SVector<1>)> h_) : h(h_) {}

        //setter for the nonlinear function
        void set_nonlinearity(std::function<double(SVector<N>, SVector<1>)> h_) {h = h_;}
    }; // end of NonLinearityBase
        

} // end namespace core
} // end namespace fdapde


#endif //FDAPDE_CORE_NON_LINEARITY_BASE_H