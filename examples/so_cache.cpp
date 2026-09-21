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

// compile from the repository root with c++ -std=c++20 -O3 -I. examples/so_cache.cpp
#include <fdaPDE/manifold_optimization.h>

#include <chrono>
#include <iostream>

int main() {
    using namespace fdapde;
    using Dense = Matrix<double, 6, 6>;
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    using Point = RotationMatrix<double, 6, 6>;
    using CachedPoint = RotationMatrix<double, 6, 6, RotationCache::Log>;
    constexpr int repetitions = 2000;
    double checksum = 0;
    Dense input {IdentityMatrix<double, 6, 6>()};
    for (int k = 0; k < 6; k += 2) {
        const double angle = .2 + .1 * k;
        input(k, k) = input(k + 1, k + 1) = std::cos(angle);
        input(k, k + 1) = -std::sin(angle);
        input(k + 1, k) = std::sin(angle);
    }
    // per-point logarithms reuse only the selected rotation cache
    {
        const Point point(input);
        const auto setup = Clock::now();
        const CachedPoint cached(point);
        const double preparation = Milliseconds(Clock::now() - setup).count();
        const auto first = rotation_log(point);
        const auto second = rotation_log(cached);
        // cached and uncached paths must agree before timing their repeated use
        fdapde_strong_assert(
          std::abs(first(0, 1) - second(0, 1)) < 1e-12, std::runtime_error, "rotation cache mismatch");
        auto start = Clock::now();
        for (int i = 0; i < repetitions; ++i) {
            const auto value = rotation_log(point);
            checksum += value(0, 1);
        }
        const double uncached = Milliseconds(Clock::now() - start).count();
        start = Clock::now();
        for (int i = 0; i < repetitions; ++i) {
            const auto value = rotation_log(cached);
            checksum += value(0, 1);
        }
        const double ready = Milliseconds(Clock::now() - start).count();
        std::cout << "rotation log: preparation " << preparation << " ms, uncached " << uncached << " ms, cached "
                  << ready << " ms\n";
    }
    // general symmetric exponentials reuse eigenpairs while retaining domain and finite-result checks
    {
        SymmetricMatrix<double, 6, 6> value;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j <= i; ++j) value(i, j) = i == j ? 1. + .1 * i : .02;
        CachedSymmetricMatrix<double, 6, 6> cached(value);
        const auto setup = Clock::now();
        (void)cached.cache();
        const double preparation = Milliseconds(Clock::now() - setup).count();
        const auto first = expm(value);
        const auto second = expm(cached);
        // caching must preserve the independently evaluated matrix exponential
        fdapde_strong_assert(
          std::abs(first(0, 0) - second(0, 0)) < 1e-12, std::runtime_error, "symmetric cache mismatch");
        auto start = Clock::now();
        for (int i = 0; i < repetitions; ++i) {
            const auto result = expm(value);
            checksum += result(0, 0);
        }
        const double uncached = Milliseconds(Clock::now() - start).count();
        start = Clock::now();
        for (int i = 0; i < repetitions; ++i) {
            const auto result = expm(cached);
            checksum += result(0, 0);
        }
        const double ready = Milliseconds(Clock::now() - start).count();
        std::cout << "symmetric exp: preparation " << preparation << " ms, uncached " << uncached << " ms, cached "
                  << ready << " ms\n";
    }
    // prepared curves and repeated exponential evaluation both certify rotation destinations
    {
        const manifold::SOGeometry<double, 6> geometry;
        const Point first = Point::Identity(), last(input);
        const auto tangent = geometry.logarithm(first, last);
        const auto setup = Clock::now();
        const auto curve = geometry.geodesic(first, last);
        const double preparation = Milliseconds(Clock::now() - setup).count();
        const Point endpoint(curve(1));
        // preparation must preserve the supplied endpoint
        fdapde_strong_assert(
          geometry.distance(endpoint, last) < 1e-10, std::runtime_error, "prepared endpoint mismatch");
        auto start = Clock::now();
        for (int i = 0; i < repetitions; ++i) {
            const auto result = geometry.exponential(first, tangent, double(i) / (repetitions - 1));
            checksum += result(0, 0);
        }
        const double unprepared = Milliseconds(Clock::now() - start).count();
        start = Clock::now();
        for (int i = 0; i < repetitions; ++i) {
            const Point result(curve(double(i) / (repetitions - 1)));
            checksum += result(0, 0);
        }
        const double prepared = Milliseconds(Clock::now() - start).count();
        std::cout << "rotation curve: preparation " << preparation << " ms, exponential " << unprepared
                  << " ms, prepared " << prepared << " ms\n";
    }
    std::cout << "repetitions " << repetitions << ", checksum " << checksum << '\n';
}
