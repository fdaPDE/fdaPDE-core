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

#include <fdaPDE/dense_linear_algebra.h>

int main() {
    using Matrix = fdapde::Matrix<double, 2, 2>;
    fdapde::GMRES<Matrix, fdapde::IdentityPreconditioner<Matrix>> solver(fdapde::IdentityPreconditioner<Matrix> {});
    Matrix rhs;
    // solve rejects a matrix right-hand side instead of silently processing one column
    solver.solve(rhs);
}
