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

#ifndef __HESSIAN_H__
#define __HESSIAN_H__

#include "matrix_field.h"
#include "scalar_field.h"

namespace fdapde {

template <typename Derived_> class Hessian : public fdapde::MatrixBase<Derived_::StaticInputSize, Hessian<Derived_>> {
   public:
    using Derived = Derived_;
    template <typename T> using Meta = Hessian<T>;
    using Base = MatrixBase<Derived::StaticInputSize, Hessian<Derived>>;
    using FunctorType = PartialDerivative<std::decay_t<Derived>, 2>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::StaticInputSize;
    static constexpr int Cols = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;
    using Base::operator();

    explicit constexpr Hessian(const Derived& xpr) : Base(), derived_(xpr), xpr_(), data_() {
        if constexpr (StaticInputSize == Dynamic) data_.resize(xpr_.input_size(), xpr_.input_size(), xpr_.input_size());
        for (int i = 0; i < xpr.input_size(); ++i) {
            for (int j = 0; j <= i; ++j) { data_(i, j) = PartialDerivative<std::decay_t<Derived>, 2>(xpr, i, j); }
        }
	xpr_ = data_.template symmetric_view<fdapde::Lower>();
    }
    // getters
    constexpr const PartialDerivative<std::decay_t<Derived>, 2>& operator()(int i, int j) { return xpr_(i, j); }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return xpr_.eval(i, j, p); }
    template <typename InputType, typename Dest> constexpr void eval_at(const InputType& p, Dest& dest) const {
        xpr_.eval_at(p, dest);
	return;
    }
    constexpr int rows() const { return data_.rows(); }
    constexpr int cols() const { return data_.cols(); }
    constexpr int input_size() const { return data_.input_size(); }
    constexpr int size() const { return data_.size(); }
    constexpr const Derived& derived() const { return derived_; }
   private:
    MatrixField<StaticInputSize, Rows, Cols, FunctorType> data_;
    MatrixSymmetricView<MatrixField<StaticInputSize, Rows, Cols, FunctorType>, fdapde::Lower> xpr_;
    typename internals::ref_select<Derived>::type derived_;
};

}   // namespace fdapde

#endif // __HESSIAN_H__
