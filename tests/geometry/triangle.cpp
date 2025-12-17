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
//
//

<<<<<<<< HEAD:tests/geometry/triangle.cpp
#include <fdaPDE/geometry.h>
#include <gtest/gtest.h>   // testing framework
using namespace fdapde;
========
#ifndef __FDAPDE_EXECUTION_TYPE_H__
#define __FDAPDE_EXECUTION_TYPE_H__
>>>>>>>> 023997a3 (Implemented multithreading support across FDAPDE):fdaPDE/src/multithreading/execution_type.h

TEST(geometry, triangle) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(60, 60);

<<<<<<<< HEAD:tests/geometry/triangle.cpp
    // std::cout << D.n_nodes() << std::endl;

    
    EXPECT_DOUBLE_EQ(1.0 / D.n_cells(), D.cell(0).measure());


    std::cout << D.cell(2000).barycenter() << std::endl;

    // try point location with r_tree
    Matrix<double, 1, 2> pts;
    pts.row(0) = D.cell(2000).barycenter().transpose();
    std::cout << D.locate(pts)[0] << std::endl;
  
}
========
namespace execution {
    struct execution_parallel {}; 
    inline constexpr execution_parallel par {};
}

#endif
>>>>>>>> 023997a3 (Implemented multithreading support across FDAPDE):fdaPDE/src/multithreading/execution_type.h
