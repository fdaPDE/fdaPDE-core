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

#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>

// Availability skeleton only; do not add this source to CMake before the P4-E native MdArray boundary is restored.
// Sources: 86ff6d12:fdaPDE/src/linear_algebra/mdarray.h and repaired evidence
// 9c39f3f:test/linear_algebra/{native_mdarray.cpp,native_mdarray_no_debug.cpp}.
// TODO(P4-E): cover checked shape/indexing, const and mutable views, copy/move/resize, row/column-major rectangular
// storage, invalidation rules, and FDAPDE_NO_DEBUG without restoring BinaryMap or an implicit Eigen bridge.
