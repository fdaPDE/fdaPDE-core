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

#include <fdaPDE/execution.h>

#include <gtest/gtest.h>   // testing framework
using namespace fdapde;

TEST(execution, task_graphs) {

    int x = 0, y = 0;
    std::vector<int> v1(1000);
    
    TaskGraph g;
    auto n1 = g.add_node([&] { x = 1; });
    auto n2 = g.add_node([&] { y = 2; });
    auto n3 = g.add_node([&] { fdapde::parallel_for(0, v1.size(), [&](int i) { v1[i] = x + y; }); });
    n3.succeeds(n1, n2);
    
    fdapde::parallel_execute(g);
    
    for(int x : v1) { EXPECT_EQ(x, 3); }
}
