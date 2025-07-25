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

#ifndef __FDAPDE_OPTIMIZATION_MODULE_H__
#define __FDAPDE_OPTIMIZATION_MODULE_H__

// clang-format off

// include required modules
#include "linear_algebra.h"    // pull Eigen first
#include "utility.h"
#include "fields.h"

namespace fdapde {

template <typename Opt> struct is_gradient_based_opt {
   private:
    using Opt_ = std::decay_t<Opt>;
    static constexpr int N = Opt_::static_input_size;
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
   public:
    static constexpr bool value = !Opt_::gradient_free && requires(Opt_ opt) {
        { opt.x_old    } -> std::convertible_to<vector_t>;
        { opt.x_new    } -> std::convertible_to<vector_t>;
        { opt.update   } -> std::convertible_to<vector_t>;
        { opt.grad_old } -> std::convertible_to<vector_t>;
        { opt.grad_new } -> std::convertible_to<vector_t>;
    };
};
template <typename Opt> static constexpr bool is_gradient_based_opt_v = is_gradient_based_opt<Opt>::value;

template <typename Opt> struct is_gradient_free_opt {
   private:
    using Opt_ = std::decay_t<Opt>;
    static constexpr int N = Opt_::static_input_size;
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
   public:
    static constexpr bool value = Opt_::gradient_free && requires(Opt_ opt) {
        { opt.x_curr   } -> std::convertible_to<vector_t>;
	{ opt.obj_curr } -> std::convertible_to<double>;
    };
};
template <typename Opt> static constexpr bool is_gradient_free_opt_v = is_gradient_free_opt<Opt>::value;
  
}   // namespace fdapde

// callbacks
#include "src/optimization/callbacks.h"
#include "src/optimization/backtracking.h"
#include "src/optimization/wolfe.h"

// algorithms
#include "src/optimization/grid_search.h"
#include "src/optimization/newton.h"
#include "src/optimization/gradient_descent.h"
#include "src/optimization/conjugate_gradient.h"
#include "src/optimization/bfgs.h"
#include "src/optimization/lbfgs.h"
#include "src/optimization/nelder_mead.h"

// clang-format on

#endif   // __FDAPDE_OPTIMIZATION_MODULE_H__
