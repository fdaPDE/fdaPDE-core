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

#define FDAPDE_NO_DEBUG
#include <fdaPDE/linear_algebra.h>

using oversized_mdarray_extents = fdapde::MdExtents<50000, 50000>;
static_assert(sizeof(oversized_mdarray_extents) > 0);

int main() { return 0; }
