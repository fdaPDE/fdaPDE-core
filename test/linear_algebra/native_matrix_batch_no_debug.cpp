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

#include <fdaPDE/linear_algebra.h>

#include <array>
#include <span>
#include <stdexcept>

namespace native = fdapde::linalg;

using Batch = native::MatrixBatchView<double, 2, 3>;

int main() {
    std::array<double, 37> malformed {};
    try {
        const Batch batch {std::span<double>(malformed)};
        static_cast<void>(batch);
        return 1;
    } catch (const std::invalid_argument&) { }

    std::array<double, 12> storage {};
    Batch batch {std::span<double>(storage)};
    for (const int index : {-1, 2}) {
        try {
            static_cast<void>(batch[index]);
            return 2;
        } catch (const std::out_of_range&) { }
    }

    std::span<double> empty_storage;
    Batch empty {empty_storage};
    if (!empty.empty() || empty.size() != 0) return 3;
    try {
        static_cast<void>(empty[0]);
        return 4;
    } catch (const std::out_of_range&) { }

    batch[1](1, 2) = 7.0;
    return storage[11] == 7.0 ? 0 : 5;
}
