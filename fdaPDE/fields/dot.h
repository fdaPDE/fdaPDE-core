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

template <int StaticInputSize, typename Derived> struct MatrixBase;

template <typename Lhs, typename Rhs> class DotProduct : public ScalarBase<Lhs::StaticInputSize, DotProduct<Lhs, Rhs>> {
    fdapde_static_assert(
      (Lhs::StaticInputSize == Dynamic || Rhs::StaticInputSize == Dynamic ||
       Lhs::StaticInputSize == Rhs::StaticInputSize) &&
        ((Lhs::Cols == 1 &&
          ((Rhs::Cols == 1 && Lhs::Rows == Rhs::Rows) || (Rhs::Rows == 1 && Lhs::Rows == Rhs::Cols))) ||
        (Lhs::Rows == 1 &&
          ((Rhs::Rows == 1 && Lhs::Cols == Rhs::Cols) || (Rhs::Cols == 1 && Lhs::Rows == Rhs::Rows)))),
      INVALID_OPERAND_SIZES_FOR_DOT_PRODUCT);
    fdapde_static_assert(
      std::is_same_v<typename Lhs::InputType FDAPDE_COMMA typename Rhs::InputType>,
      YOU_MIXED_MATRIX_FIELDS_WITH_DIFFERENT_INPUT_TYPE);
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = DotProduct<T1, T2>;
    using Base = ScalarBase<LhsDerived::StaticInputSize, DotProduct<Lhs, Rhs>>;
    using InputType = typename LhsDerived::InputType;
    using Scalar = decltype(std::declval<typename LhsDerived::Scalar>() * std::declval<typename RhsDerived::Scalar>());
    static constexpr int StaticInputSize = LhsDerived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = LhsDerived::XprBits | RhsDerived::XprBits;

    constexpr DotProduct(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) {
        if constexpr (LhsDerived::Cols == Dynamic || RhsDerived::Cols == Dynamic) {
            fdapde_assert(
              (lhs.cols() == 1 &&
               ((rhs.cols() == 1 && lhs_.rows() == rhs_.rows()) || (rhs.rows() == 1 && lhs_.rows() == rhs_.cols()))) ||
              (lhs.rows() == 1 &&
               ((rhs.rows() == 1 && lhs.cols() == rhs.cols()) || (rhs.cols() == 1 && lhs.cols() == rhs.rows()))));
        }
        if constexpr (LhsDerived::StaticInputSize == Dynamic || RhsDerived::StaticInputSize == Dynamic) {
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
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
   private:
    typename internals::ref_select<const LhsDerived>::type lhs_;
    typename internals::ref_select<const RhsDerived>::type rhs_;
};

template <typename Lhs, typename Rhs>
constexpr DotProduct<Lhs, Rhs> dot(
  const fdapde::MatrixBase<Lhs::StaticInputSize, Lhs>& lhs, const fdapde::MatrixBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return DotProduct<Lhs, Rhs>(lhs.derived(), rhs.derived());
}

// integration with Eigen types
  
namespace internals {

template <typename Lhs, typename Rhs, typename FieldType_ = std::conditional_t<is_eigen_dense_v<Lhs>, Rhs, Lhs>>
class dot_product_eigen_impl : public ScalarBase<FieldType_::StaticInputSize, dot_product_eigen_impl<Lhs, Rhs>> {
    using FieldType = std::conditional_t<is_eigen_dense_v<Lhs>, Rhs, Lhs>;
    using EigenType = std::conditional_t<is_eigen_dense_v<Lhs>, Lhs, Rhs>;
    static constexpr bool is_field_lhs = std::is_same_v<FieldType, Lhs>;
    fdapde_static_assert(
      (FieldType::Cols == 1 &&
       ((EigenType::ColsAtCompileTime == 1 && FieldType::Rows == EigenType::RowsAtCompileTime) ||
        (EigenType::RowsAtCompileTime == 1 && FieldType::Rows == EigenType::ColsAtCompileTime))) ||
      (FieldType::Rows == 1 &&
       ((EigenType::RowsAtCompileTime == 1 && FieldType::Cols == EigenType::ColsAtCompileTime) ||
        (EigenType::ColsAtCompileTime == 1 && FieldType::Cols == EigenType::RowsAtCompileTime))),
      INVALID_OPERAND_SIZES_FOR_DOT_PRODUCT);
   public:
    using LhsDerived = Lhs;
    using RhsDerived = Rhs;
    template <typename T1, typename T2> using Meta = dot_product_eigen_impl<T1, T2>;
    using Base = ScalarBase<FieldType::StaticInputSize, dot_product_eigen_impl<Lhs, Rhs>>;
    using InputType = typename FieldType::InputType;
    using Scalar = decltype(std::declval<typename FieldType::Scalar>() * std::declval<typename EigenType::Scalar>());
    static constexpr int StaticInputSize = FieldType::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = FieldType::XprBits;

    dot_product_eigen_impl(const Lhs& lhs, const Rhs& rhs) : Base(), lhs_(lhs), rhs_(rhs) {
        if constexpr (
          FieldType::Cols == Dynamic || EigenType::ColsAtCompileTime == Dynamic || FieldType::Rows == Dynamic ||
          EigenType::RowsAtCompileTime == Dynamic) {
            fdapde_assert(
              (lhs.cols() == 1 &&
               ((rhs.cols() == 1 && lhs.rows() == rhs.rows()) || (rhs.rows() == 1 && lhs.rows() == rhs.cols()))) ||
              (lhs.rows() == 1 &&
               ((rhs.rows() == 1 && lhs.cols() == rhs.cols()) || (rhs.cols() == 1 && lhs.cols() == rhs.rows()))));
        }
    }
    Scalar operator()(const InputType& p) const {
        Scalar dot_ = 0;
        int n = lhs_.cols() == 1 ? lhs_.rows() : lhs_.cols();
        for (int i = 0; i < n; ++i) {
	    if constexpr (is_field_lhs) dot_ += lhs_.eval(i, p) * rhs_[i];
	    else dot_ += lhs_[i] * rhs_.eval(i, p);
	}
        return dot_;
    }
    constexpr int input_size() const {
        if constexpr (is_field_lhs)  return lhs_.input_size();
        else return rhs_.input_size();
    }
    constexpr const LhsDerived& lhs() const { return lhs_; }
    constexpr const RhsDerived& rhs() const { return rhs_; }
   protected:
    std::conditional_t<is_eigen_dense_v<Lhs>, const Lhs&, typename internals::ref_select<const Lhs>::type> lhs_;
    std::conditional_t<is_eigen_dense_v<Rhs>, const Rhs&, typename internals::ref_select<const Rhs>::type> rhs_;
};

}   // namespace internals

template <typename Lhs, typename Rhs>
struct DotProduct<Lhs, Eigen::MatrixBase<Rhs>> : public internals::dot_product_eigen_impl<Lhs, Rhs> {
    DotProduct(const Lhs& lhs, const Rhs& rhs) : internals::dot_product_eigen_impl<Lhs, Rhs>(lhs, rhs) { }
};
template <typename Lhs, typename Rhs>
struct DotProduct<Eigen::MatrixBase<Lhs>, Rhs> : public internals::dot_product_eigen_impl<Lhs, Rhs> {
    DotProduct(const Lhs& lhs, const Rhs& rhs) : internals::dot_product_eigen_impl<Lhs, Rhs>(lhs, rhs) { }
};

template <typename Lhs, typename Rhs>
constexpr DotProduct<Lhs, Eigen::MatrixBase<Rhs>>
dot(const fdapde::MatrixBase<Lhs::StaticInputSize, Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs) {
    return DotProduct<Lhs, Eigen::MatrixBase<Rhs>>(lhs.derived(), rhs.derived());
}
template <typename Lhs, typename Rhs>
constexpr DotProduct<Eigen::MatrixBase<Lhs>, Rhs>
dot(const Eigen::MatrixBase<Lhs>& lhs, const fdapde::MatrixBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return DotProduct<Eigen::MatrixBase<Lhs>, Rhs>(lhs.derived(), rhs.derived());
}

}   // namespace fdapde

#endif   // __DOT_H__
