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

// compile from the repository root with c++ -std=c++20 -O3 -I. examples/spd_batch_interpolation_le.cpp
#include <fdaPDE/manifold_optimization.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

// compares five interpolation paths with setup timed and result checks outside timing
int main() {
    using namespace fdapde;
    using namespace fdapde::manifold;
    using Dense = Matrix<double, 2, 2>;
    using Symmetric = SymmetricMatrix<double, 2>;
    using Point = SPDMatrix<double, 2>;
    using CachedPoint = SPDMatrix<double, 2, Cache::Log>;
    using Geometry = LogEuclideanSPDGeometry<double, 2>;
    using CachedGeometry = LogEuclideanSPDGeometry<double, 2, Usage::Distance>;
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    constexpr std::size_t point_count = 10001;
    constexpr std::size_t repetitions = 9;
    const Dense first_data({2.0, 0.5, 0.5, 3.0});
    const Dense last_data({5.0, -1.0, -1.0, 4.0});
    const std::array<const char*, 5> names {
      "nessuna cache", "cache ingressi e risultato temporaneo", "cache ingressi, risultato NoCache",
      "geodetica preparata, risultato SPD verificato", "geodetica preparata, simmetrica (verifica fuori tempo)"};
    std::array<std::array<double, repetitions>, 5> samples {};
    std::array<double, 5> errors {};
    double checksum = 0;

    // run zero warms up all paths; the following nine runs contribute to the medians
    for (std::size_t run = 0; run <= repetitions; ++run) {
        std::array<double, 5> elapsed {};
        std::array<MatrixBatch<Symmetric>, 5> results;

        // 1. nessuna cache
        {
            const auto start = Clock::now();
            const Geometry geometry;
            const Point first(first_data), last(last_data);
            const auto tangent = geometry.logarithm(first, last);
            MatrixBatch<Point> points(point_count);
            for (std::size_t i = 0; i < point_count; ++i) {
                const double step = static_cast<double>(i) / (point_count - 1);
                points[i] = geometry.exponential(first, tangent, step);
            }
            elapsed[0] = Milliseconds(Clock::now() - start).count();
            results[0] = MatrixBatch<Symmetric>(points);
        }

        // 2. cache sugli ingressi e sul risultato temporaneo
        {
            const auto start = Clock::now();
            const CachedGeometry geometry;
            const CachedPoint first(first_data), last(last_data);
            const auto tangent = geometry.logarithm(first, last);
            MatrixBatch<Point> points(point_count);
            for (std::size_t i = 0; i < point_count; ++i) {
                const double step = static_cast<double>(i) / (point_count - 1);
                points[i] = geometry.exponential(first, tangent, step);
            }
            elapsed[1] = Milliseconds(Clock::now() - start).count();
            results[1] = MatrixBatch<Symmetric>(points);
        }

        // 3. cache sugli ingressi, risultato senza cache
        {
            const auto start = Clock::now();
            const Geometry geometry;
            const CachedPoint first(first_data), last(last_data);
            const auto tangent = geometry.logarithm(first, last);
            MatrixBatch<Point> points(point_count);
            for (std::size_t i = 0; i < point_count; ++i) {
                const double step = static_cast<double>(i) / (point_count - 1);
                points[i] = geometry.exponential(first, tangent, step);
            }
            elapsed[2] = Milliseconds(Clock::now() - start).count();
            results[2] = MatrixBatch<Symmetric>(points);
        }

        // 4. geodetica preparata tramite l'API, con verifica SPD di ogni risultato
        {
            const auto start = Clock::now();
            const Geometry geometry;
            const CachedPoint first(first_data), last(last_data);
            const auto curve = geometry.geodesic(first, last);
            MatrixBatch<Point> points(point_count);
            for (std::size_t i = 0; i < point_count; ++i) {
                const double step = static_cast<double>(i) / (point_count - 1);
                points[i] = curve(step);
            }
            elapsed[3] = Milliseconds(Clock::now() - start).count();
            results[3] = MatrixBatch<Symmetric>(points);
        }

        // 5. stessa espressione del caso 4, con verifica SPD soltanto dopo il cronometro
        {
            const auto start = Clock::now();
            const Geometry geometry;
            const CachedPoint first(first_data), last(last_data);
            const auto curve = geometry.geodesic(first, last);
            MatrixBatch<Symmetric> points(point_count);
            for (std::size_t i = 0; i < point_count; ++i) {
                const double step = static_cast<double>(i) / (point_count - 1);
                points[i] = curve(step);
            }
            elapsed[4] = Milliseconds(Clock::now() - start).count();
            results[4] = MatrixBatch<Symmetric>(points);
        }

        // verify every result against the uncached path, after all five timers have stopped
        for (std::size_t test = 0; test < results.size(); ++test) {
            for (std::size_t i = 0; i < point_count; ++i) {
                const Point checked(results[test][i]);
                double difference = 0, norm = 0;
                for (int row = 0; row < 2; ++row)
                    for (int col = 0; col < 2; ++col) {
                        const double expected = results[0][i](row, col);
                        const double delta = checked(row, col) - expected;
                        difference += delta * delta;
                        norm += expected * expected;
                        if (i == 0 || i == point_count - 1) {
                            const double endpoint = (i == 0 ? first_data : last_data)(row, col);
                            // the first and last sampled matrices must reproduce the supplied endpoints
                            fdapde_strong_assert(
                              std::abs(checked(row, col) - endpoint) < 1e-10, std::runtime_error,
                              "incorrect interpolation endpoint");
                        }
                    }
                const double error = std::sqrt(difference / norm);
                // all coefficients must agree with the public LE path in relative Frobenius norm
                fdapde_strong_assert(
                  std::isfinite(error) && error < 1e-10, std::runtime_error,
                  "interpolation result differs from reference");
                errors[test] = std::max(errors[test], error);
                checksum += checked(0, 0) + 2 * checked(1, 0) + checked(1, 1);
            }
            if (run != 0) samples[test][run - 1] = elapsed[test];
        }
    }

    for (auto& sample : samples) std::sort(sample.begin(), sample.end());
    std::cout << point_count << " punti; setup e allocazione inclusi; mediana di " << repetitions
              << " ripetizioni dopo warmup; casi in ordine fisso; verifiche fuori tempo\n";
    for (std::size_t test = 0; test < names.size(); ++test) {
        const double median = samples[test][repetitions / 2];
        std::cout << names[test] << ": " << std::fixed << std::setprecision(3) << median << " ms; "
                  << samples[0][repetitions / 2] / median << "x baseline; errore max " << std::scientific
                  << errors[test] << '\n';
    }
    std::cout << "speedup preparata SPD / cache con temporaneo: " << std::fixed << std::setprecision(3)
              << samples[1][repetitions / 2] / samples[3][repetitions / 2] << "x\n";
    std::cout << "rapporto preparata verificata / verifica esclusa: "
              << samples[3][repetitions / 2] / samples[4][repetitions / 2] << "x\n";
    std::cout << "checksum: " << std::setprecision(12) << checksum << '\n';
}
