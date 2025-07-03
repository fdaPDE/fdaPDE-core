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

#ifndef __FDAPDE_GEOFRAME_MODULE_H__
#define __FDAPDE_GEOFRAME_MODULE_H__

// clang-format off

// include required modules
#include "linear_algebra.h"    // pull Eigen first
#include "utility.h"
#include "geometry.h"
#include "io.h"

// define basic symbols
namespace fdapde {

[[maybe_unused]] constexpr int MESH_NODES = 0;
inline struct areal_layer_tag {} gf_areal;
inline struct point_layer_tag {} gf_point;
enum class ltype : int { point = 0, areal = 1 };
  
namespace internals {
  enum class dtype : int { flt64 = 0, flt32 = 1, int64 = 2, int32 = 3, bin = 4, str = 5 };    // runtime data type id
}   // namespace internals

namespace data_t {

inline struct flt64_ : std::type_identity<double      > { internals::dtype type_id = internals::dtype::flt64; } flt64;
inline struct flt32_ : std::type_identity<float       > { internals::dtype type_id = internals::dtype::flt32; } flt32;
inline struct int64_ : std::type_identity<std::int64_t> { internals::dtype type_id = internals::dtype::int64; } int64;
inline struct int32_ : std::type_identity<std::int32_t> { internals::dtype type_id = internals::dtype::int32; } int32;
inline struct bin_   : std::type_identity<bool        > { internals::dtype type_id = internals::dtype::bin;   } bin;
inline struct str_   : std::type_identity<std::string > { internals::dtype type_id = internals::dtype::str;   } str;
  
}   // namespace data_t

namespace internals {

template <typename T> constexpr auto dtype_from_static_type() {
    using T_ = std::decay_t<T>;
    if constexpr (std::is_same_v<T_, double      >) return data_t::flt64;
    if constexpr (std::is_same_v<T_, float       >) return data_t::flt32;
    if constexpr (std::is_same_v<T_, std::int64_t>) return data_t::int64;
    if constexpr (std::is_same_v<T_, std::int32_t>) return data_t::int32;
    if constexpr (std::is_same_v<T_, bool        >) return data_t::bin;
    if constexpr (std::is_same_v<T_, std::string >) return data_t::str;
}

template <typename FieldDType_, typename F_, typename... Args>
    requires(std::is_same_v<FieldDType_, internals::dtype>)
void dispatch_to_dtype(const FieldDType_& field_dtype, F_&& f, Args&&... args) {
    using dtypes =
      std::tuple<data_t::flt64_, data_t::flt32_, data_t::int64_, data_t::int32_, data_t::bin_, data_t::str_>;
    std::apply(
      [&](const auto&... ts) {
          (
            [&]() {
                if (field_dtype == ts.type_id) {
                    f.template operator()<typename std::decay_t<decltype(ts)>::type>(std::forward<Args>(args)...);
                }
            }(),
            ...);
      },
      dtypes {});
}

template <typename F_, typename... Args> void foreach_dtype(F_&& f, Args&&... args) {
    using dtypes =
      std::tuple<data_t::flt64_, data_t::flt32_, data_t::int64_, data_t::int32_, data_t::bin_, data_t::str_>;
    std::apply(
      [&](const auto&... ts) {
          (f.template operator()<typename std::decay_t<decltype(ts)>::type>(std::forward<Args>(args)...), ...);
      },
      dtypes {});
}
  
}   // namespace internals
}   // namespace fdapde

// data structure
#include "src/geoframe/data_layer.h"
#include "src/geoframe/areal_layer.h"
#include "src/geoframe/point_layer.h"
#include "src/geoframe/geo_layer.h"
#include "src/geoframe/geoframe.h"

// clang-format on

#endif   // __FDAPDE_FIELDS_MODULE_H__
