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

#ifndef __FDAPDE_CORE_NON_LINEARITY_H__
#define __FDAPDE_CORE_NON_LINEARITY_H__

#include <type_traits>

#include "../utils/symbols.h"
#include "scalar_expressions.h"

#include <fdaPDE/finite_elements.h>
// using fdapde::core::LagrangianElement;

namespace fdapde{
    namespace core{

    /* template <int N, typename B>
    class NonLinearReactionBase {
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
        NonLinearReactionBase() = default;
        NonLinearReactionBase(std::function<double(SVector<N>, SVector<1>)> h_) : h(h_) {}

        //setter for the nonlinear function
        void set_nonlinearity(std::function<double(SVector<N>, SVector<1>)> h_) {h = h_;}
    }; // end of NonLinearReactionBase

    template <int N, typename B>
    class NonLinearReaction : public NonLinearReactionBase<N, B>,
                              public ScalarExpr<N, NonLinearReaction<N, B>>  {
    public:
        typedef std::shared_ptr<DVector<double>> VecP;
        
        auto operator()(VecP f_prev) const{
            this->f_prev_ = f_prev;
            return *this;
        }

        double operator()(const SVector<N>& x) const{
            return this->h(x, this->f(x));
        }
    }; // end of NonLinearReaction

    template <int N, typename B>
    class NonLinearReactionPrime: public NonLinearReactionBase<N, B>,
                                  public ScalarExpr<N, NonLinearReactionPrime<N, B>>  {
    public:
        typedef std::shared_ptr<DVector<double>> VecP;
    
        double operator()(const SVector<N>& x) const{
            std::function<double(SVector<1>)> lambda_fun = [&] (SVector<1> ff) -> double {return this->h(x, ff);};
            ScalarField<1> lambda_field(lambda_fun);
            SVector<1> au;
            au << this->f(x);
            return lambda_field.derive()(au)[0] * this->f(x)[0];    // per newton method
        }

        auto operator()(VecP f_prev) const{
            this->f_prev_ = f_prev;
            return *this;
        }
    }; // end of NonLinearReactionPrime */
        

    } // end namespace core
} // end namespace fdapde


#endif //FDAPDE_CORE_NON_LINEARITY_H