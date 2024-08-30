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

#ifndef __SPACE_TIME_FIELD_H__
#define __SPACE_TIME_FIELD_H__

#include "scalar_field.h"

namespace fdapde {

// a SpaceTimeField is a ScalarField with an implicit time dimension
template <int Size, typename FunctorType_ = std::function<double(static_dynamic_vector_selector_t<Size>, double)>>
class SpaceTimeField : public ScalarBase<Size, SpaceTimeField<Size, FunctorType_>> {
  using FunctorType = std::decay_t<FunctorType_>;
  using traits = fn_ptr_traits<&FunctorType::operator()>;
  using TimeCoordinateType = std::tuple_element_t<1, typename traits::ArgsType>;
  fdapde_static_assert(
    traits::n_args == 2 && std::is_arithmetic_v<TimeCoordinateType>,
    PROVIDED_FUNCTOR_MUST_ACCEPT_EXACTLY_TWO_ARGUMENTS_OR_TIME_DIMENSION_IS_NOT_OF_ARITHMETIC_TYPE);
 public:
  using Base = ScalarBase<Size, ScalarField<Size, FunctorType>>;
  using InputType = std::tuple_element_t<0, typename traits::ArgsType>;
  using Scalar = typename std::invoke_result<FunctorType, InputType, TimeCoordinateType>::type;
  static constexpr int StaticInputSize = Size;
  static constexpr int NestAsRef = 1;
  static constexpr int XprBits = 0;

  constexpr SpaceTimeField() requires(StaticInputSize != Dynamic) : f_() { }
  explicit SpaceTimeField(int n) requires(StaticInputSize == Dynamic)
      : Base(), f_(), dynamic_input_size_(n) { }
  constexpr explicit SpaceTimeField(const FunctorType& f) : f_(f) { }

  // assignment from lambda expression
  template <typename LamdaType> SpaceTimeField& operator=(const LamdaType& lambda) {
      fdapde_static_assert(
        std::is_same_v<FunctorType FDAPDE_COMMA std::function<Scalar(InputType, TimeCoordinateType)>> &&
          std::is_invocable_v<LamdaType FDAPDE_COMMA InputType FDAPDE_COMMA TimeCoordinateType> &&
          std::is_convertible_v<typename std::invoke_result<
            LamdaType FDAPDE_COMMA InputType FDAPDE_COMMA TimeCoordinateType>::type FDAPDE_COMMA Scalar>,
        INVALID_SCALAR_FUNCTION_ASSIGNMENT);
      f_ = lambda;
      return *this;
  }
  constexpr int input_size() const { return StaticInputSize == Dynamic ? dynamic_input_size_ : StaticInputSize; }
  // evaluation at point
  constexpr Scalar operator()(const InputType& x) const { return f_(x, t_); }
  constexpr Scalar operator()(const InputType& x) { return f_(x, t_); }
  constexpr Scalar operator()(const InputType& x, const TimeCoordinateType& t) const { return f_(x, t); }
  constexpr Scalar operator()(const InputType& x, const TimeCoordinateType& t) { return f_(x, t); }
  // fix time coordinate
  constexpr SpaceTimeField& at(TimeCoordinateType t) {
      t_ = t;
      return *this;
  }
  void resize(int dynamic_input_size) {
      fdapde_static_assert(StaticInputSize == Dynamic, YOU_CALLED_A_DYNAMIC_METHOD_ON_A_STATIC_SIZED_FIELD);
      dynamic_input_size_ = dynamic_input_size;
  }
private:
  int dynamic_input_size_ = 0;   // run-time base space dimension
  FunctorType f_;
  TimeCoordinateType t_ = 0;
};

}   // namespace fdapde

#endif   // __SPACE_TIME_FIELD_H__
