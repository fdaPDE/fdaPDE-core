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

#ifndef __BILINEAR_FORM_TRAITS_H__
#define __BILINEAR_FORM_TRAITS_H__

#include <tuple>
#include <type_traits>

#include "../../utils/traits.h"
#include "dt.h"
#include "gradient.h"
#include "identity.h"
#include "laplacian.h"

namespace fdapde {
namespace core {

// trait to detect if the bilinear form is symmetric.
template <typename E> struct is_symmetric {
    static constexpr bool value = std::decay<E>::type::is_symmetric;
};

// trait to detect if the bilinear form denotes a parabolic PDE.
template <typename E_> struct is_parabolic {
    typedef typename std::decay<E_>::type E;
    // returns true if the time derivative operator dT() is detected in the expression
    static constexpr bool value = has_type<dT, decltype(std::declval<E>().get_operator_type())>::value;
};

}   // namespace core
}   // namespace fdapde

#endif   // __BILINEAR_FORM_TRAITS_H__
