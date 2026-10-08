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

// compile from the repository root with c++ -std=c++20 -O2 -pthread -I.
// examples/linear_algebra/08_boolean_and_multidimensional_storage.cpp -o /tmp/08_boolean_and_multidimensional_storage
#include <fdaPDE/dense_linear_algebra.h>

#include <array>
#include <iostream>

using namespace fdapde;

// select measurements with Boolean masks and organize several small tables in one array
int main() {
    // 1. comparisons create compact Boolean matrices, with one bit per coefficient
    const Matrix<double, 2, 3> readings({1., 4., 2., 5., 3., 6.});
    Matrix<bool, 2, 3> selected(readings.cwise() > 3.);
    const Matrix<bool, 2, 3> complement(~selected);
    const Matrix<bool, 2, 3> either(selected | complement);
    std::cout << "1. Readings:\n"
              << readings << "\nMask for readings above three:\n"
              << selected << "\nSelected entries: " << selected.count() << "\nComplement:\n"
              << complement << "\nTheir union selects everything: " << std::boolalpha << either.all() << '\n';

    // 2. packed Boolean views bind the packed words returned by data(), not a plain bool buffer
    MatrixView<bool, 2, 3> bit_view(selected.data());
    bit_view(0, 0) = true;
    Vector<bool, Dynamic> active(3, true);
    active.resize(5);
    std::cout << "Mask after selecting the first entry through a view:\n"
              << selected << "\nGrowing a flag vector keeps its old bits and clears new ones:\n"
              << active << '\n';

    // 3. organize values along experiment, sensor and time axes with fixed extents
    MdArray<double, MdExtents<2, 2, 3>> readings_by_experiment;
    for (int experiment = 0; experiment < 2; ++experiment)
        for (int sensor = 0; sensor < 2; ++sensor)
            for (int time = 0; time < 3; ++time)
                readings_by_experiment(experiment, sensor, time) = 100. * experiment + 10. * sensor + time;

    // 4. slice<0> fixes the experiment axis and exposes a borrowed sensor-by-time table
    auto second = readings_by_experiment.slice<0>(1);
    // type: MdView<MdArray<double, MdExtents<2, 2, 3>>, 2>
    std::cout << "4. Second experiment (sensor rows, time columns):\n";
    for (int sensor = 0; sensor < 2; ++sensor) {
        for (int time = 0; time < 3; ++time) std::cout << second(sensor, time) << ' ';
        std::cout << '\n';
    }

    // 5. blocks select inclusive intervals; full_extent keeps the whole sensor axis
    auto destination = readings_by_experiment.block(1, full_extent, std::pair {1, 2});
    // type: MdView<MdArray<double, MdExtents<2, 2, 3>>, 3>
    auto source = readings_by_experiment.block(1, full_extent, std::pair {0, 1});
    // type: MdView<MdArray<double, MdExtents<2, 2, 3>>, 3>
    destination = source;
    std::cout << "Shift the first two times into the last two slots, sensor zero: ";
    for (int time = 0; time < 3; ++time) std::cout << second(0, time) << ' ';
    std::cout << '\n';

    // 6. choose a runtime sensor count and column-major storage without changing logical indexing
    MdArray<double, MdExtents<Dynamic, 3>, ColMajor> table(2);
    table = second;

    // 7. borrow an external buffer with MdMap and copy the table through a full block
    std::array<double, 6> buffer {};
    MdMap<double, MdExtents<2, 3>> map(buffer.data());
    auto mapped_values = map.block(full_extent, full_extent);
    // type: MdView<MdMap<double, MdExtents<2, 3>>, 2>
    mapped_values = table;
    auto last_time = map.col(2);
    // type: MdView<MdMap<double, MdExtents<2, 3>>, 2>
    last_time(1, 0) = -1.;
    std::cout << "7. External buffer after copying the table and editing its last column:";
    for (double value : buffer) std::cout << ' ' << value;
    std::cout << '\n';
}
