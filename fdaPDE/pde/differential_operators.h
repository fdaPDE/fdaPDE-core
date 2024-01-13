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

#ifndef __DIFFERENTIAL_OPERATORS_H__
#define __DIFFERENTIAL_OPERATORS_H__

#include <type_traits>
#include "../utils/traits.h"

namespace fdapde {
namespace core {
  
// macro for the definition of a differential operator symbol, with its factory method
#define FDAPDE_DEFINE_DIFFERENTIAL_OPERATOR(OP, NAME)                                                                  \
    template <typename T, typename... Ts> class OP { };                                                                \
    template <typename T, typename... Ts> OP<T, Ts...> NAME(Ts... f) { return OP<T, Ts...>(std::forward<Ts>(f)...); }

// supported differential operators. The specific discretization is left to strategy T
FDAPDE_DEFINE_DIFFERENTIAL_OPERATOR(Diffusion,   diffusion  );
FDAPDE_DEFINE_DIFFERENTIAL_OPERATOR(Laplacian,   laplacian  );
FDAPDE_DEFINE_DIFFERENTIAL_OPERATOR(Advection,   advection  );
FDAPDE_DEFINE_DIFFERENTIAL_OPERATOR(Reaction,    reaction   );
FDAPDE_DEFINE_DIFFERENTIAL_OPERATOR(dT,          dt         );
FDAPDE_DEFINE_DIFFERENTIAL_OPERATOR(BiLaplacian, bilaplacian);

// trait to detect if the bilinear form obtained from the weak formulation of a differential operator is symmetric.
template <typename E> struct is_symmetric {
    static constexpr bool value = std::decay<E>::type::is_symmetric;
};
// trait to detect if the differential operator denotes a parabolic problem.
template <typename E_> struct is_parabolic {
    typedef typename std::decay<E_>::type E;
    static constexpr bool value = has_instance_of<dT, decltype(std::declval<E>().get_operator_type())>::value;
};
// detects if the problem is stationary (does not involve time)
template <typename E_> struct is_stationary {
    typedef typename std::decay<E_>::type E;
    static constexpr bool value = !has_instance_of<dT, decltype(std::declval<E>().get_operator_type())>::value;
};
  
}   // namespace core
}   // namespace fdapde

#endif   // __DIFFERENTIAL_OPERATORS_H__
