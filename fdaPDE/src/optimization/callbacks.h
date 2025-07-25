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

#ifndef __FDAPDE_OPTIMIZATION_CALLBACKS_H__
#define __FDAPDE_OPTIMIZATION_CALLBACKS_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

template <typename Hook, typename... Args>
bool opt_hooks_loop(Hook hook, std::tuple<Args...>& callbacks) {
    bool b = false;
    std::apply([&](auto&&... callback) { ([&]() { b |= hook(callback); }(), ...); }, callbacks);
    return b;
}

template <typename Opt, typename Obj, typename... Args>
bool exec_eval_hooks(Opt& optimizer, Obj& objective, std::tuple<Args...>& callbacks) {
    return opt_hooks_loop(
      [&](auto&& callback) {
          if constexpr (requires(std::decay_t<decltype(callback)> c, Opt opt, Obj obj) {
                            { c.eval_hook(opt, obj) } -> std::same_as<bool>;
                        }) {
              return callback.eval_hook(optimizer, objective);
          }
	  return false;
      },
      callbacks);
}

template <typename Opt, typename Obj, typename... Args>
bool exec_grad_hooks(Opt& optimizer, Obj& objective, std::tuple<Args...>& callbacks) {
    return opt_hooks_loop(
      [&](auto&& callback) {
          if constexpr (requires(std::decay_t<decltype(callback)> c, Opt opt, Obj obj) {
                            { c.grad_hook(opt, obj) } -> std::same_as<bool>;
                        }) {
              return callback.grad_hook(optimizer, objective);
          }
	  return false;
      },
      callbacks);
}

template <typename Opt, typename Obj, typename... Args>
bool exec_adapt_hooks(Opt& optimizer, Obj& objective, std::tuple<Args...>& callbacks) {
    return opt_hooks_loop(
      [&](auto&& callback) {
          if constexpr (requires(std::decay_t<decltype(callback)> c, Opt opt, Obj obj) {
                            { c.adapt_hook(opt, obj) } -> std::same_as<bool>;
                        }) {
              return callback.adapt_hook(optimizer, objective);
          }
	  return false;
      },
      callbacks);
}

template <typename Opt, typename Obj> bool exec_stop_if(Opt& optimizer, Obj& objective) {
    bool b = false;
    if constexpr (requires(Opt opt, Obj obj) {
                      { obj.stop_if(opt) } -> std::same_as<bool>;
                  }) {
        b |= objective.stop_if(optimizer);
    }
    return b;
}

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_OPTIMIZATION_CALLBACKS_H__
