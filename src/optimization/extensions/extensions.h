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

#ifndef __EXTENSION_H__
#define __EXTENSION_H__

#include "../../utils/traits.h"

namespace fdapde {
namespace core {

// define detection-idiom based extension traits
define_has(pre_update_step);
define_has(post_update_step);

template <typename Opt, typename Obj, typename... Args>
bool execute_pre_update_step(Opt& optimizer, Obj& objective, Args&... args) {
    bool b = false;
    (   // fold expand parameter pack
      [&] {
          if constexpr (has_pre_update_step<Args, bool(Opt&, Obj&)>::value) {
              b |= args.pre_update_step(optimizer, objective);
          }
      }(),
      ...);
    return b;
}

template <typename Opt, typename Obj, typename... Args>
bool execute_post_update_step(Opt& optimizer, Obj& objective, Args&... args) {
    bool b = false;
    (   // fold expand parameter pack
      [&] {
          if constexpr (has_post_update_step<Args, bool(Opt&, Obj&)>::value) {
              b |= args.post_update_step(optimizer, objective);
          }
      }(),
      ...);
    return b;
}

}   // namespace core
}   // namespace fdapde

#endif   // __EXTENSION_H__
