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
#include <type_traits>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
#    define FDAPDE_BENCH_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#    define FDAPDE_BENCH_NOINLINE __declspec(noinline)
#else
#    define FDAPDE_BENCH_NOINLINE
#endif

namespace {

using namespace fdapde;

/// @brief evaluates the public product assignment in a separate call, retaining its source snapshot and allocation
template <typename Result, typename Lhs, typename Rhs>
FDAPDE_BENCH_NOINLINE void native_product(Result& result, const Lhs& lhs, const Rhs& rhs) {
    result = lhs * rhs;
}

/// @brief reports the median duration of five equally sized rounds in nanoseconds per public product assignment
template <typename Operation> double median_time(Operation operation, int repetitions) {
    std::array<double, 5> times;
    for (double& time : times) {
        const auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < repetitions; ++repeat) operation();
        time = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / repetitions;
    }
    std::sort(times.begin(), times.end());
    return times[2];
}

/// @brief initializes exact binary fractions and compares the public product with an ordered scalar reference in
/// release
template <typename Lhs, typename Rhs, typename Result>
bool measure_case(const char* name, Lhs lhs, Rhs rhs, Result result, int repetitions) {
    using Scalar = std::common_type_t<typename Lhs::Scalar, typename Rhs::Scalar>;
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int k = 0; k < lhs.cols(); ++k) lhs(i, k) = Scalar((7 * i + 3 * k) % 17 - 8) / Scalar(4);
    }
    for (int k = 0; k < rhs.rows(); ++k) {
        for (int j = 0; j < rhs.cols(); ++j) rhs(k, j) = Scalar((5 * k + 11 * j) % 19 - 9) / Scalar(8);
    }
    std::vector<Scalar> expected(static_cast<std::size_t>(lhs.rows()) * rhs.cols(), Scalar(0));
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < rhs.cols(); ++j) {
            for (int k = 0; k < lhs.cols(); ++k) expected[i * rhs.cols() + j] += lhs(i, k) * rhs(k, j);
        }
    }
    native_product(result, lhs, rhs);
    const double nanoseconds = median_time([&] { native_product(result, lhs, rhs); }, repetitions);
    if (result.rows() != lhs.rows() || result.cols() != rhs.cols()) {
        std::fprintf(
          stderr, "%s returned shape %dx%d, expected %dx%d\n", name, result.rows(), result.cols(), lhs.rows(),
          rhs.cols());
        return false;
    }
    double checksum = 0;
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            const Scalar reference = expected[i * rhs.cols() + j];
            if (result(i, j) != reference) {
                std::fprintf(
                  stderr, "%s coefficient (%d,%d): expected %.17g, got %.17g\n", name, i, j,
                  static_cast<double>(reference), static_cast<double>(result(i, j)));
                return false;
            }
            checksum += static_cast<double>(result(i, j));
        }
    }
    std::printf(
      "%s %s %c%c%c %d %d %d %d %.3f %.12g\n", name, std::is_same_v<Scalar, float> ? "float" : "double",
      Lhs::StorageOrder == RowMajor ? 'R' : 'C', Rhs::StorageOrder == RowMajor ? 'R' : 'C',
      Result::StorageOrder == RowMajor ? 'R' : 'C', lhs.rows(), lhs.cols(), rhs.cols(), repetitions, nanoseconds,
      checksum);
    return true;
}

/// @brief creates separate dynamic operand/output allocations before measuring public product assignment
template <typename Scalar, int LhsOrder, int RhsOrder, int ResultOrder>
bool dynamic_case(const char* name, int rows, int inner, int cols, int repetitions) {
    return measure_case(
      name, Matrix<Scalar, Dynamic, Dynamic, LhsOrder>(rows, inner),
      Matrix<Scalar, Dynamic, Dynamic, RhsOrder>(inner, cols),
      Matrix<Scalar, Dynamic, Dynamic, ResultOrder>(rows, cols), repetitions);
}

/// @brief measures unaligned externally owned operands and output while retaining public product snapshot costs
bool view_case() {
    constexpr int Rows = 65, Inner = 97, Cols = 33;
    std::vector<double> lhs(Rows * Inner + 2), rhs(Inner * Cols + 2), result(Rows * Cols + 2);
    return measure_case(
      "odd_views", MatrixView<double, Dynamic, Dynamic>(lhs.data() + 1, Rows, Inner),
      MatrixView<double, Dynamic, Dynamic>(rhs.data() + 1, Inner, Cols),
      MatrixView<double, Dynamic, Dynamic>(result.data() + 1, Rows, Cols), 25);
}

}   // namespace

/// @brief benchmarks native public dense products with scalar-oracle checks active independently of ndebug
int main() {
#if defined(__clang__)
    std::printf("compiler clang %s\n", __clang_version__);
#elif defined(__GNUC__)
    std::printf("compiler gcc %s\n", __VERSION__);
#endif
#ifdef FDAPDE_NO_DEBUG
    std::printf("fdapde assertions disabled\n");
#else
    std::printf("fdapde assertions enabled\n");
#endif
    std::printf("case scalar lhs_rhs_output rows inner cols repetitions median_ns checksum\n");
    // fixed fem-sized products include stack snapshots while dynamic cases include heap snapshot allocations
    return !(
      measure_case("fem3", Matrix<double, 3, 3>(), Matrix<double, 3, 3>(), Matrix<double, 3, 3>(), 20000) &&
      measure_case(
        "fem4", Matrix<double, 4, 4, ColMajor>(), Matrix<double, 4, 4, ColMajor>(), Matrix<double, 4, 4, ColMajor>(),
        20000) &&
      measure_case("fem10", Matrix<double, 10, 10>(), Matrix<double, 10, 10>(), Matrix<double, 10, 10>(), 2000) &&
      dynamic_case<double, RowMajor, RowMajor, RowMajor>("odd", 65, 97, 33, 25) &&
      dynamic_case<double, RowMajor, RowMajor, RowMajor>("square_row", 128, 128, 128, 5) &&
      dynamic_case<double, ColMajor, ColMajor, ColMajor>("square_col", 128, 128, 128, 5) &&
      dynamic_case<double, RowMajor, ColMajor, RowMajor>("mixed_row", 128, 128, 128, 5) &&
      dynamic_case<double, ColMajor, RowMajor, ColMajor>("mixed_col", 128, 128, 128, 5) &&
      dynamic_case<double, RowMajor, RowMajor, RowMajor>("rectangular", 96, 257, 17, 10) &&
      dynamic_case<float, RowMajor, RowMajor, RowMajor>("square_float", 128, 128, 128, 5) && view_case());
}
