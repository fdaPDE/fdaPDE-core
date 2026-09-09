// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <fdaPDE/linear_algebra.h>

int main() {
    fdapde::Matrix<double, 2, 2> matrix;
    const auto reshaped = matrix.template reshape<1, 3>();
    return reshaped.size();
}
