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

#ifndef __FE_ASSEMBLER_H__
#define __FE_ASSEMBLER_H__

#include "fe_scalar_assembler.h"
#include "fe_vector_assembler.h"

namespace fdapde {

template <typename Triangulation, int Options, typename... Quadrature> class FeAssembler {
    fdapde_static_assert(
      sizeof...(Quadrature) < 2, YOU_CAN_PROVIDE_AT_MOST_ONE_QUADRATURE_FORMULA_TO_A_FE_CELL_ASSEMBLER);
    std::tuple<Quadrature...> quadrature_;
    std::conditional_t<
      Options == CellMajor, typename Triangulation::cell_iterator, typename Triangulation::boundary_iterator>
      begin_, end_;
   public:
    FeAssembler() = default;
    template <typename Iterator>
    FeAssembler(const Iterator& begin, const Iterator& end, const Quadrature&... quadrature) :
        begin_(begin), end_(end), quadrature_(std::make_tuple(quadrature...)) { }

    template <typename Form> auto operator()(const Form& form) const {
        static constexpr bool has_trial_space = meta::xpr_find<
          decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; }), std::decay_t<Form>>();
        static constexpr bool has_test_space = meta::xpr_find<
          decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; }), std::decay_t<Form>>();

        if constexpr (has_trial_space && has_test_space) {   // bilinear form discretization
            if constexpr (sizeof...(Quadrature) == 0) {
                return internals::fe_matrix_scalar_assembly_loop<Triangulation, Form, Options> {form, begin_, end_};
            } else {
                return internals::fe_matrix_scalar_assembly_loop<
                  Triangulation, Form, Options, std::tuple_element_t<0, std::tuple<Quadrature...>>> {
                  form, begin_, end_, std::get<0>(quadrature_)};
            }
        } else {   // functional (forcing-term like) discretization
            if constexpr (sizeof...(Quadrature) == 0) {
                return internals::fe_vector_scalar_assembly_loop<Triangulation, Form, Options> {form, begin_, end_};
            } else {
                return internals::fe_vector_scalar_assembly_loop<
                  Triangulation, Form, Options, std::tuple_element_t<0, std::tuple<Quadrature...>>> {
                  form, begin_, end_, std::get<0>(quadrature_)};
            }
        }
    }
};

template <typename Triangulation, typename... Quadrature>
using FeCellAssembler = FeAssembler<Triangulation, CellMajor, Quadrature...>;
template <typename Triangulation, typename... Quadrature>
using FeFaceAssembler = FeAssembler<Triangulation, FaceMajor, Quadrature...>;

template <typename Triangulation, typename... Quadrature>
auto integral(const Triangulation& triangulation, Quadrature... quadrature) {
    return FeCellAssembler<Triangulation, Quadrature...>(
      triangulation.cells_begin(), triangulation.cells_end(), quadrature...);
}
template <typename Triangulation, typename... Quadrature>
auto integral(
  const CellIterator<Triangulation>& begin, const CellIterator<Triangulation>& end, Quadrature... quadrature) {
    return FeCellAssembler<Triangulation, Quadrature...>(begin, end, quadrature...);
}
// boundary integration
template <typename Triangulation, typename... Quadrature>
auto integral(
  const BoundaryIterator<Triangulation>& begin, const BoundaryIterator<Triangulation>& end, Quadrature... quadrature) {
    return FeFaceAssembler<Triangulation, Quadrature...>(begin, end, quadrature...);
}
template <typename Triangulation, typename... Quadrature>
auto integral(
  const std::pair<BoundaryIterator<Triangulation>, BoundaryIterator<Triangulation>>& range, Quadrature... quadrature) {
    return FeFaceAssembler<Triangulation, Quadrature...>(range.first, range.second, quadrature...);
}
  
}   // namespace fdapde

#endif   // __FE_ASSEMBLER_H__
