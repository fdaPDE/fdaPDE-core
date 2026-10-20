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

#include <fdaPDE/finite_elements.h>

#include <chrono>
#include <iomanip>
#include <iostream>

/// @brief measures scalar P1 enumeration and physical coordinate extraction on growing tetrahedral grids
int main() {
    using Clock = std::chrono::steady_clock;
    using namespace fdapde;
    std::cout << "nodes,cells,enumeration_seconds,coordinate_seconds,coordinate_error\n" << std::setprecision(9);
    for (int resolution : {12, 20, 28}) {
        const auto mesh = Triangulation<3, 3>::UnitCube(resolution);
        const auto begin = Clock::now();
        const FeSpace space(mesh, P1<1>);
        const auto enumerated = Clock::now();
        const auto coordinates = space.dof_handler().dofs_coords();
        const auto end = Clock::now();
        const double error = (coordinates - mesh.nodes()).norm();
        // geometric node coordinates independently verify that timing preserves the P1 result
        fdapde_strong_assert(error < 1e-11, std::runtime_error, "P1 coordinates differ from geometric nodes");
        std::cout << mesh.n_nodes() << ',' << mesh.n_cells() << ','
                  << std::chrono::duration<double>(enumerated - begin).count() << ','
                  << std::chrono::duration<double>(end - enumerated).count() << ',' << error << std::endl;
    }
}
