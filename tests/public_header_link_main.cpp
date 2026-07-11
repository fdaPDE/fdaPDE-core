// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <fdaPDE/geometry.h>

int public_header_link_other();

int main() {
    fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> point(1, 2);
    point(0, 0) = 0.0;
    point(0, 1) = 0.0;
    const auto boundary = fdapde::hexagonal_lattice_boundary(point, 1.0);
    return boundary.rows() == 6 && public_header_link_other() > 0 ? 0 : 1;
}
