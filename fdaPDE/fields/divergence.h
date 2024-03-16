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

#ifndef FDAPDE_CORE_DIVERGENCE_H
#define FDAPDE_CORE_DIVERGENCE_H

namespace fdapde{
namespace core{

// forward declaration
template <int M, int N> class DiscretizedVectorField;

template <int M, int N, typename VectorExprType> class Divergence: public ScalarExpr<M, Divergence<M, N, VectorExprType>> {
   private:
    using This = Divergence<M, N, VectorExprType>;
    using InnerVectorType = typename static_dynamic_vector_selector<M>::type;
    VectorExprType op_;
public:
    Divergence(const VectorExprType& op)  : op_(op) { }
    inline double operator()(const InnerVectorType& x) const {
        double result = 0;
        for(int i = 0; i < op_.inner_size(); ++i){
            result += op_[i].derive()[i](x);
        }
        return result;
    }
    template <typename T> const This& forward(T i) {
        op_.forward(i);
        return *this;
    }
};

template <int M, int N>
class Divergence<M, N, DiscretizedVectorField<M, N>> : public ScalarExpr<M, Divergence<M, N, DiscretizedVectorField<M, N>>> {
   private:
    using This = Divergence<M, N, DiscretizedVectorField<M, N>>;
    using InnerVectorType = typename static_dynamic_vector_selector<M>::type;
    DVector<double> data_;
    double value_ = 0;
   public:
    Divergence(const DVector<double>& data)  : data_(data) { fdapde_assert(data.size() != 0); }
    inline double operator()(const InnerVectorType& x) const { return value_; }
    template <typename T> const This& forward(T i) {
        value_ = data_.row(i);
        return *this;
    }
};

template <typename VectorExprType>
auto div(const VectorExprType& expr) {
    return expr.div();
}
template <> auto div(const SVector<2>& expr) {
    return 0.0;
}

}   // namespace core
}   // namespace fdapde

#endif //FDAPDE_CORE_DIVERGENCE_H
