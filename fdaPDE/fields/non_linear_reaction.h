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

#ifndef __FDAPDE_CORE_NON_LINEAR_REACTION_H__
#define __FDAPDE_CORE_NON_LINEAR_REACTION_H__

#include <type_traits>

#include "../utils/symbols.h"
#include "scalar_expressions.h"
#include "non_linearity_base.h"
#include "field_derivatives.h"

namespace fdapde{
namespace core{

    template <int N, typename B>
    class NonLinearReaction : public NonLinearityBase<N, B>,
                              public ScalarExpr<N, NonLinearReaction<N, B>>  {
    public:
        typedef std::shared_ptr<DVector<double>> VecP;
        
        NonLinearReaction() = default;
        NonLinearReaction(std::function<double(SVector<N>, SVector<1>)> h_) : NonLinearityBase<N,B>(h_){}

        auto operator()(VecP f_prev) const{
            this->f_prev_ = f_prev;
            return *this;
        }

        double operator()(const SVector<N>& x) const{
            return this->h(x, this->f(x));
        }
    }; // end of NonLinearReaction

    template <int N, typename B>
    class NonLinearReactionPrime: public NonLinearityBase<N, B>,
                                  public ScalarExpr<N, NonLinearReactionPrime<N, B>>  {
    public:
        typedef std::shared_ptr<DVector<double>> VecP;
    
        NonLinearReactionPrime() = default;
        NonLinearReactionPrime(std::function<double(SVector<N>, SVector<1>)> h_) : NonLinearityBase<N,B>(h_){}

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
    }; // end of NonLinearReactionPrime
        

} // end namespace core
} // end namespace fdapde


#endif //FDAPDE_CORE_NON_LINEAR_REACTION_H