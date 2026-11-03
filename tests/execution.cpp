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
#include <gtest/gtest.h>

#include "execution/adaptive_dispatcher.cpp"
#include "execution/chase_lev_queue.cpp"
#include "execution/future.cpp"
#include "execution/mpsc_queue.cpp"
#include "execution/parallel_for.cpp"
#include "execution/parallel_for_each.cpp"
#include "execution/parallel_reduce.cpp"
#include "execution/static_dispatcher.cpp"
#include "execution/task_batch.cpp"
#include "execution/task_graph.cpp"
#include "execution/task_handle.cpp"
#include "execution/threaded_executor.cpp"

int main(int argc, char** argv) {
    fdapde::parallel_set_num_threads(4);
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    fdapde::parallel_join();
    return result;
}
