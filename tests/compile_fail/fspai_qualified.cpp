// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <fdaPDE/sparse_linear_algebra.h>

// rejects coefficients outside the unqualified floating-point public contract
int main() { fdapde::FSPAI<const double> invalid; }
