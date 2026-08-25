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

#include <Eigen/SparseCore>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using native_triplet = fdapde::Triplet<double>;
using eigen_sparse = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

std::vector<native_triplet> make_grid_triplets(int subdivisions) {
    const int nodes_per_side = subdivisions + 1;
    std::vector<native_triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(18 * subdivisions * subdivisions));
    const auto append_triangle = [&triplets](int first, int second, int third) {
        const int nodes[3] {first, second, third};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                triplets.emplace_back(nodes[row], nodes[col], row == col ? 2.0 : -0.5);
            }
        }
    };
    for (int row = 0; row < subdivisions; ++row) {
        for (int col = 0; col < subdivisions; ++col) {
            const int lower_left = row * nodes_per_side + col;
            const int lower_right = lower_left + 1;
            const int upper_left = lower_left + nodes_per_side;
            const int upper_right = upper_left + 1;
            append_triangle(lower_left, lower_right, upper_right);
            append_triangle(lower_left, upper_right, upper_left);
        }
    }
    return triplets;
}

double median(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

struct observation {
    int nonzeros = 0;
    std::uint64_t checksum = 0;

    friend bool operator==(const observation&, const observation&) = default;
};

std::uint64_t mix(std::uint64_t hash, int row, int col, double value) {
    hash ^= static_cast<std::uint64_t>(row) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    hash ^= static_cast<std::uint64_t>(col) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    hash ^= std::bit_cast<std::uint64_t>(value) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

template <typename Function> double measure_once(Function& function, observation& observed) {
    const auto start = clock_type::now();
    observed = function();
    const auto stop = clock_type::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

bool benchmark_case(int subdivisions, int repetitions) {
    const int nodes = (subdivisions + 1) * (subdivisions + 1);
    const auto native_triplets = make_grid_triplets(subdivisions);
    std::vector<Eigen::Triplet<double, int>> eigen_triplets;
    eigen_triplets.reserve(native_triplets.size());
    for (const auto& triplet : native_triplets) {
        eigen_triplets.emplace_back(triplet.row(), triplet.col(), triplet.value());
    }

    auto native = [&] {
        const fdapde::SparseMatrix<double> matrix(nodes, nodes, native_triplets);
        observation result {matrix.non_zeros(), 0};
        for (int row = 0; row < matrix.rows(); ++row) {
            for (const auto entry : matrix.row(row)) {
                result.checksum = mix(result.checksum, row, entry.column(), entry.value());
            }
        }
        return result;
    };
    auto eigen = [&] {
        eigen_sparse matrix(nodes, nodes);
        matrix.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
        matrix.prune(0.0);
        matrix.makeCompressed();
        observation result {static_cast<int>(matrix.nonZeros()), 0};
        for (int outer = 0; outer < matrix.outerSize(); ++outer) {
            for (eigen_sparse::InnerIterator entry(matrix, outer); entry; ++entry) {
                result.checksum = mix(result.checksum, entry.row(), entry.col(), entry.value());
            }
        }
        return result;
    };

    observation native_observation;
    observation eigen_observation;
    for (int warmup = 0; warmup < 3; ++warmup) {
        static_cast<void>(native());
        static_cast<void>(eigen());
    }
    std::vector<double> native_samples;
    std::vector<double> eigen_samples;
    native_samples.reserve(static_cast<std::size_t>(repetitions));
    eigen_samples.reserve(static_cast<std::size_t>(repetitions));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        if (repetition % 2 == 0) {
            native_samples.push_back(measure_once(native, native_observation));
            eigen_samples.push_back(measure_once(eigen, eigen_observation));
        } else {
            eigen_samples.push_back(measure_once(eigen, eigen_observation));
            native_samples.push_back(measure_once(native, native_observation));
        }
    }
    const double native_ms = median(native_samples);
    const double eigen_ms = median(eigen_samples);

    if (native_observation != eigen_observation) {
        std::cerr << "sparse benchmark structure mismatch: native=" << native_observation.nonzeros
                  << " eigen=" << eigen_observation.nonzeros << '\n';
        return false;
    }
    const double ratio = native_ms / eigen_ms;
    const std::string_view verdict = ratio <= 1.10 ? "pass" : ratio <= 1.25 ? "profile" : "block";
    std::cout << std::fixed << std::setprecision(3) << "subdivisions=" << subdivisions << " nodes=" << nodes
              << " raw_triplets=" << native_triplets.size() << " nonzeros=" << native_observation.nonzeros
              << " native_median_ms=" << native_ms << " eigen_median_ms=" << eigen_ms << " ratio=" << ratio
              << " verdict=" << verdict << '\n';
    return verdict != "block";
}

}   // namespace

int main() { return benchmark_case(32, 9) && benchmark_case(128, 5) ? 0 : 1; }
