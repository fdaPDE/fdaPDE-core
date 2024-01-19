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
bool execute_pre_update_step(Opt& optimizer, Obj& objective, std::tuple<Args...>& callbacks) {
    bool b = false;
    auto exec_callback = [&](auto&& callback) {
        if constexpr (has_pre_update_step<std::decay_t<decltype(callback)>, bool(Opt&, Obj&)>::value) {
            b |= callback.pre_update_step(optimizer, objective);
        }
    };
    std::apply([&](auto&&... callback) { (exec_callback(callback), ...); }, callbacks);
    return b;
}

template <typename Opt, typename Obj, typename... Args>
bool execute_post_update_step(Opt& optimizer, Obj& objective, std::tuple<Args...>& callbacks) {
    bool b = false;
    auto exec_callback = [&](auto&& callback) {
        if constexpr (has_post_update_step<std::decay_t<decltype(callback)>, bool(Opt&, Obj&)>::value) {
            b |= callback.post_update_step(optimizer, objective);
        }
    };
    std::apply([&](auto&&... callback) { (exec_callback(callback), ...); }, callbacks);
    return b;
}

}   // namespace core
}   // namespace fdapde

#endif   // __EXTENSION_H__
