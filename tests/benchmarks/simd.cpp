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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
#    define FDAPDE_BENCH_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#    define FDAPDE_BENCH_NOINLINE __declspec(noinline)
#else
#    define FDAPDE_BENCH_NOINLINE
#endif

using Vec = fdapde::Vector<double, fdapde::Dynamic>;
using Mat = fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>;
using ColMat = fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic, fdapde::ColMajor>;
using View = fdapde::MatrixView<double, fdapde::Dynamic, fdapde::Dynamic>;

// separate calls preserve repeated public assignments in optimized benchmark builds
extern "C" FDAPDE_BENCH_NOINLINE void vector_scale(Vec& x, double a) { x *= a; }
extern "C" FDAPDE_BENCH_NOINLINE void matrix_scale(Mat& x, double a) { x *= a; }
extern "C" FDAPDE_BENCH_NOINLINE void column_matrix_scale(ColMat& x, double a) { x *= a; }
extern "C" FDAPDE_BENCH_NOINLINE void view_scale(View& x, double a) { x *= a; }
extern "C" FDAPDE_BENCH_NOINLINE void vector_affine(Vec& out, const Vec& x, const Vec& y, double a, double b) {
    out = a * x + b * y;
}
extern "C" FDAPDE_BENCH_NOINLINE void matrix_affine(Mat& out, const Mat& x, const Mat& y, double a, double b) {
    out = a * x + b * y;
}
extern "C" FDAPDE_BENCH_NOINLINE void view_affine(View& out, const View& x, const View& y, double a, double b) {
    out = a * x + b * y;
}

// report the median of five equally sized rounds in nanoseconds per public call
template <typename Operation> double median_time(Operation operation, int repetitions) {
    std::array<double, 5> times;
    for (double& time : times) {
        const auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < repetitions; ++repeat) { operation(); }
        time = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / repetitions;
    }
    std::sort(times.begin(), times.end());
    return times[2];
}

// verify every coefficient against scalar arithmetic even when release builds define ndebug
template <typename Operation>
bool measure(
  const char* name, Operation operation, int repetitions, const double* output, int size, double expected,
  double& checksum) {
    const double time = median_time(operation, repetitions);
    for (int i = 0; i < size; ++i) {
        if (output[i] != expected) {
            std::fprintf(stderr, "%s coefficient %d: expected %.17g, got %.17g\n", name, i, expected, output[i]);
            return false;
        }
        checksum += output[i];
    }
    std::printf("%s %.3f ns/call\n", name, time);
    return true;
}

int main() {
    for (const int size : {3, 51, 12288, 393216}) {
        const int repetitions = size < 20000 ? 2000 : 200;
        const double scale = 1.000001;
        double scaled_value = 1.25;
        for (int repeat = 0; repeat < 5 * repetitions; ++repeat) { scaled_value *= scale; }

        Vec x(size, 1.25), y(size, 2.0), z(size, 1.25), out(size);
        Mat mx(size / 3, 3, 1.25), my(size / 3, 3, 2.0), mz(size / 3, 3, 1.25), mout(size / 3, 3);
        ColMat cmx(3, size / 3, 1.25);
        std::vector<double> vx(size, 1.25), vy(size, 2.0), vz(size, 1.25), vo(size);
        View xv(vx.data(), size / 3, 3), yv(vy.data(), size / 3, 3), zv(vz.data(), size / 3, 3),
          ov(vo.data(), size / 3, 3);
        double checksum = 0.0;
        std::printf("n=%d repetitions=%d\n", size, repetitions);

        // compare all scaling outputs with the same number of scalar multiplications
        if (
          !measure(
            "vector_scale", [&] { vector_scale(x, scale); }, repetitions, x.data(), size, scaled_value, checksum) ||
          !measure(
            "matrix_scale", [&] { matrix_scale(mx, scale); }, repetitions, mx.data(), size, scaled_value, checksum) ||
          !measure(
            "column_matrix_scale", [&] { column_matrix_scale(cmx, scale); }, repetitions, cmx.data(), size,
            scaled_value, checksum) ||
          !measure(
            "view_scale", [&] { view_scale(xv, scale); }, repetitions, vx.data(), size, scaled_value, checksum)) {
            return 1;
        }

        // distinct input allocations retain the public affine assignment's snapshot cost
        const double affine_value = 0.5 * 2.0 + 0.25 * 1.25;
        if (
          !measure(
            "vector_affine", [&] { vector_affine(out, y, z, 0.5, 0.25); }, repetitions, out.data(), size, affine_value,
            checksum) ||
          !measure(
            "matrix_affine", [&] { matrix_affine(mout, my, mz, 0.5, 0.25); }, repetitions, mout.data(), size,
            affine_value, checksum) ||
          !measure(
            "view_affine", [&] { view_affine(ov, yv, zv, 0.5, 0.25); }, repetitions, vo.data(), size, affine_value,
            checksum)) {
            return 1;
        }
        std::printf("checksum %.17g\n", checksum);
    }
}
