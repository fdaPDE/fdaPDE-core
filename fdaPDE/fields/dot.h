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

#ifndef __DOT_H__
#define __DOT_H__

#include "scalar_field.h"

namespace fdapde {

template <typename Lhs, typename Rhs> class DotProduct : public ScalarBase<Lhs::StaticInputSize, DotProduct<Lhs, Rhs>> {
    fdapde_static_assert(
      (Lhs::StaticInputSize == Dynamic || Rhs::StaticInputSize == Dynamic ||
       Lhs::StaticInputSize == Rhs::StaticInputSize) ||
        (Lhs::Cols == Dynamic || Rhs::Cols == Dynamic ||
         (Lhs::Cols == 1 &&
          ((Rhs::Cols == 1 && Lhs::Rows == Rhs::Rows) || (Rhs::Rows == 1 && Lhs::Rows == Rhs::Cols))) ||
         (Lhs::Rows == 1 && (Rhs::Rows == 1 && Lhs::Cols == Rhs::Cols) || (Rhs::Cols == 1 && Lhs::Cols == Rhs::Cols))),
      INVALID_OPERAND_SIZES_FOR_DOT_PRODUCT);
    fdapde_static_assert(
      std::is_same_v<typename Lhs::InputType FDAPDE_COMMA typename Rhs::InputType>,
      YOU_MIXED_MATRIX_FIELDS_WITH_DIFFERENT_INPUT_TYPE);
   public:
    using Base = ScalarBase<Lhs::StaticInputSize, DotProduct<Lhs, Rhs>>;
    using InputType = typename Lhs::InputType;
    using Scalar = decltype(std::declval<typename Lhs::Scalar>() * std::declval<typename Rhs::Scalar>());
    static constexpr int StaticInputSize = Lhs::StaticInputSize;
    static constexpr int NestAsRef = 0;

    constexpr DotProduct(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) {
        if constexpr (Lhs::Cols == Dynamic || Rhs::Cols == Dynamic) {
            fdapde_assert(
              (lhs.cols() == 1 &&
               ((rhs.cols() == 1 && lhs_.rows() == rhs_.rows()) || (rhs.rows() == 1 && lhs_.rows() == rhs_.cols()))) ||
              (lhs.rows() == 1 &&
               ((rhs.rows() == 1 && lhs.cols() == rhs.cols()) || (rhs.cols() == 1 && lhs.cols() == rhs.rows()))));
        }
        if constexpr (Lhs::StaticInputSize == Dynamic || Rhs::StaticInputSize == Dynamic) {
            fdapde_assert(lhs_.input_size() == rhs_.input_size());
        }
    }
    constexpr Scalar operator()(const InputType& p) const {
        Scalar dot_ = 0;
	int n = lhs_.cols() == 1 ? lhs_.rows() : lhs_.cols();
        for (int i = 0; i < n; ++i) dot_ += lhs_.eval(i, p) * rhs_.eval(i, p);
        return dot_;
    }
    constexpr int input_size() const { return lhs_.input_size(); }
    template <typename... Args> constexpr DotProduct<Lhs, Rhs>& forward(Args&&... args) {
        lhs_.forward(std::forward<Args>(args)...);
        rhs_.forward(std::forward<Args>(args)...);
        return *this;
    }
   private:
    typename internals::ref_select<const Lhs>::type lhs_;
    typename internals::ref_select<const Rhs>::type rhs_;
};

template <typename Lhs, typename Rhs> constexpr DotProduct<Lhs, Rhs> dot(const Lhs& lhs, const Rhs& rhs) {
    return DotProduct<Lhs, Rhs>(lhs, rhs);
}

}   // namespace fdapde

#endif   // __DOT_H__
