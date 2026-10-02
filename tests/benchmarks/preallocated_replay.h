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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef FDAPDE_BENCH_PREALLOCATED_REPLAY_H
#define FDAPDE_BENCH_PREALLOCATED_REPLAY_H

#include <fdaPDE/sparse_linear_algebra.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fdapde_bench {

/// @brief stores one zero-based physical or scaled metric entry independently of either backend
struct ReplayTriplet {
    int row;
    int col;
    double value;
};

/// @brief owns captured physical inputs before backend-specific storage is prepared
struct ReplayInput {
    int size = 0;
    std::vector<ReplayTriplet> omega;
    std::vector<double> c, warm, weight;
};

/// @brief owns common scaled inputs and persistent scratch buffers for fresh FISTA runs
struct ReplayWorkspace {
    int size = 0;
    std::vector<ReplayTriplet> omega, scaled_triplets;
    std::vector<double> scaled_c, inverse_diagonal, initial_u, gamma;
    std::vector<double> u, y, next, gradient, absolute_product, weight, expected_weight;
    double lipschitz = 0.0;
    double tolerance = 0.0;
    double positive_c_max = 0.0;
};

/// @brief reports one FISTA-only replay with verification and normalization outside full_ns
struct ReplayResult {
    double full_ns = 0.0;
    double spmv_ns = 0.0;
    double raw_spmv_ns = 0.0;
    double timer_overhead_ns = 0.0;
    std::uint64_t spmv_calls = 0;
    int iterations = 0;
    int restarts = 0;
    int support_count = 0;
    std::uint64_t support_hash = 14695981039346656037ULL;
    double kkt_relative = 0.0;
    double norm_error = 0.0;
    double weight_relative_error = 0.0;
    bool nonnegative = false;
    bool converged = false;
};

/// @brief reads an exact-length little-endian binary64 vector and rejects trailing or nonfinite data
inline std::vector<double> read_replay_vector(const std::string& path, int size) {
    // the captured bytes encode native IEEE binary64 coefficients
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    fdapde_strong_assert(
      std::endian::native == std::endian::little, std::runtime_error,
      "captured vectors require a little-endian binary64 host");
    std::ifstream stream(path, std::ios::binary);
    std::vector<double> values(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(double)));
    fdapde_strong_assert(
      stream && stream.peek() == std::char_traits<char>::eof(), std::runtime_error,
      "captured vector has an invalid byte count: " + path);
    fdapde_strong_assert(
      std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); }), std::runtime_error,
      "captured vector is nonfinite: " + path);
    return values;
}

/// @brief loads a general real MatrixMarket metric and its physical vectors without using Eigen
inline ReplayInput load_replay(const std::string& prefix) {
    ReplayInput input;
    std::ifstream stream(prefix + "-omega.mtx");
    std::string line, magic, object, format, scalar, symmetry;
    std::getline(stream, line);
    std::istringstream header(line);
    header >> magic >> object >> format >> scalar >> symmetry;
    fdapde_strong_assert(
      magic == "%%MatrixMarket" && object == "matrix" && format == "coordinate" && scalar == "real" &&
        symmetry == "general",
      std::runtime_error, "replay requires a full general real coordinate MatrixMarket metric");
    while (std::getline(stream, line) && (line.empty() || line[0] == '%')) { }
    std::int64_t rows = 0, cols = 0, count = 0;
    std::istringstream shape(line);
    shape >> rows >> cols >> count;
    fdapde_strong_assert(
      shape && rows > 0 && rows == cols && rows <= std::numeric_limits<int>::max() && count > 0 &&
        count <= std::numeric_limits<int>::max() && count <= rows * rows,
      std::runtime_error, "invalid replay metric shape");
    input.size = static_cast<int>(rows);
    input.omega.reserve(static_cast<std::size_t>(count));
    std::vector<std::int64_t> coordinates;
    coordinates.reserve(static_cast<std::size_t>(count));
    for (std::int64_t entry = 0; entry < count; ++entry) {
        std::int64_t row = 0, col = 0;
        double value = 0.0;
        stream >> row >> col >> value;
        fdapde_strong_assert(
          stream && row > 0 && row <= rows && col > 0 && col <= cols && std::isfinite(value), std::runtime_error,
          "invalid replay metric coordinate or coefficient");
        input.omega.push_back({static_cast<int>(row - 1), static_cast<int>(col - 1), value});
        coordinates.push_back((row - 1) * cols + col - 1);
    }
    stream >> std::ws;
    fdapde_strong_assert(
      stream.peek() == std::char_traits<char>::eof(), std::runtime_error, "replay metric contains trailing entries");
    std::sort(coordinates.begin(), coordinates.end());
    fdapde_strong_assert(
      std::adjacent_find(coordinates.begin(), coordinates.end()) == coordinates.end(), std::runtime_error,
      "replay metric contains duplicate coordinates");
    input.c = read_replay_vector(prefix + "-c.bin", input.size);
    input.warm = read_replay_vector(prefix + "-warm.bin", input.size);
    input.weight = read_replay_vector(prefix + "-weight.bin", input.size);
    return input;
}

/// @brief computes the captured physical quadratic form using compensated products and summation
inline double replay_quadratic_form(const std::vector<ReplayTriplet>& entries, const std::vector<double>& x) {
    double sum = 0.0, correction = 0.0;
    const auto accumulate = [&](double term) {
        const double next = sum + term;
        correction += std::abs(sum) >= std::abs(term) ? (sum - next) + term : (term - next) + sum;
        sum = next;
    };
    for (const auto& entry : entries) {
        if (x[entry.row] == 0.0 || x[entry.col] == 0.0) continue;
        const double product = entry.value * x[entry.row];
        const double product_error = std::fma(entry.value, x[entry.row], -product);
        const double term = product * x[entry.col];
        const double term_error = std::fma(product, x[entry.col], -term) + product_error * x[entry.col];
        accumulate(term);
        accumulate(term_error);
    }
    return sum + correction;
}

/// @brief prepares identical scaled inputs and warm seeds outside the timed kernels without certifying SPD
inline ReplayWorkspace prepare_replay(const ReplayInput& input) {
    const std::size_t size = static_cast<std::size_t>(input.size);
    fdapde_strong_assert(
      input.size > 0 && input.c.size() == size && input.warm.size() == size && input.weight.size() == size,
      std::invalid_argument, "replay input vectors must match the metric");
    ReplayWorkspace work;
    work.size = input.size;
    work.omega = input.omega;
    work.expected_weight = input.weight;
    work.scaled_triplets = input.omega;
    for (auto* buffer :
         {&work.scaled_c, &work.inverse_diagonal, &work.initial_u, &work.gamma, &work.u, &work.y, &work.next,
          &work.gradient, &work.absolute_product, &work.weight})
        buffer->resize(size);
    std::vector<double> diagonal(size, 0.0), row_sum(size, 0.0), row_count(size, 1.0);
    for (const auto& entry : input.omega) {
        fdapde_strong_assert(
          entry.row >= 0 && entry.row < input.size && entry.col >= 0 && entry.col < input.size &&
            std::isfinite(entry.value),
          std::invalid_argument, "replay metric entry is invalid");
        if (entry.row == entry.col) diagonal[entry.row] = entry.value;
    }
    double warm_max = 0.0, reference_norm2 = 0.0;
    for (int i = 0; i < input.size; ++i) {
        fdapde_strong_assert(
          diagonal[i] > 0.0 && std::isfinite(input.c[i]) && std::isfinite(input.warm[i]) &&
            std::isfinite(input.weight[i]),
          std::invalid_argument, "replay requires a positive diagonal and finite physical vectors");
        const double root = std::sqrt(diagonal[i]);
        work.inverse_diagonal[i] = 1.0 / root;
        work.scaled_c[i] = work.inverse_diagonal[i] * input.c[i];
        work.positive_c_max = std::max(work.positive_c_max, work.scaled_c[i]);
        work.initial_u[i] = root * std::max(0.0, input.warm[i]);
        fdapde_strong_assert(
          std::isfinite(work.inverse_diagonal[i]) && std::isfinite(work.scaled_c[i]) &&
            std::isfinite(work.initial_u[i]),
          std::invalid_argument, "replay diagonal scaling is nonfinite");
        warm_max = std::max(warm_max, std::abs(work.initial_u[i]));
        reference_norm2 += input.weight[i] * input.weight[i];
    }
    fdapde_strong_assert(
      reference_norm2 > 0.0 && std::isfinite(reference_norm2), std::invalid_argument,
      "replay reference weight must have a finite positive squared norm");
    for (auto& entry : work.scaled_triplets) {
        entry.value = (entry.value * work.inverse_diagonal[entry.row]) * work.inverse_diagonal[entry.col];
        row_sum[entry.row] += std::abs(entry.value);
        row_count[entry.row] += 1.0;
    }
    const double epsilon = std::numeric_limits<double>::epsilon();
    work.lipschitz = *std::max_element(row_sum.begin(), row_sum.end()) * (1.0 + std::sqrt(epsilon));
    work.tolerance = 1e-8 * work.positive_c_max;
    fdapde_strong_assert(
      work.lipschitz > 0.0 && std::isfinite(work.lipschitz) && work.positive_c_max > 0.0 &&
        std::isfinite(work.positive_c_max),
      std::invalid_argument, "replay requires a finite Lipschitz bound and a positive signal");
    for (int i = 0; i < input.size; ++i) work.gamma[i] = epsilon * row_count[i] / (1.0 - epsilon * row_count[i]);
    if (warm_max > 0.0) {
        for (double& value : work.initial_u) value /= warm_max;
        for (const auto& entry : work.scaled_triplets)
            work.gradient[entry.row] += entry.value * work.initial_u[entry.col];
        double norm2 = 0.0, signal = 0.0;
        for (int i = 0; i < input.size; ++i) {
            norm2 += work.initial_u[i] * work.gradient[i];
            signal += work.scaled_c[i] * work.initial_u[i];
        }
        fdapde_strong_assert(
          norm2 > 0.0 && std::isfinite(norm2) && std::isfinite(signal), std::invalid_argument,
          "replay warm direction has an invalid scaled norm or signal");
        const double scale = std::max(0.0, signal) / norm2;
        fdapde_strong_assert(std::isfinite(scale), std::invalid_argument, "replay warm seed scaling is nonfinite");
        for (double& value : work.initial_u) value *= scale;
    }
    return work;
}

/// @brief checks stationarity and complementarity with the production row-wise roundoff allowance
inline double replay_kkt(ReplayWorkspace& work) {
    double violation = 0.0;
    for (int i = 0; i < work.size; ++i) {
        if (!std::isfinite(work.u[i]) || !std::isfinite(work.gradient[i]))
            return std::numeric_limits<double>::infinity();
        const double residual = work.u[i] > 0.0 ? std::abs(work.gradient[i]) : std::max(0.0, -work.gradient[i]);
        violation = std::max(violation, residual);
    }
    if (violation <= work.tolerance) return violation;
    for (int i = 0; i < work.size; ++i) work.absolute_product[i] = std::abs(work.scaled_c[i]);
    for (const auto& entry : work.scaled_triplets)
        work.absolute_product[entry.row] += std::abs(entry.value) * std::abs(work.u[entry.col]);
    violation = 0.0;
    for (int i = 0; i < work.size; ++i) {
        const double residual = work.u[i] > 0.0 ? std::abs(work.gradient[i]) : std::max(0.0, -work.gradient[i]);
        violation = std::max(violation, residual - work.gamma[i] * work.absolute_product[i]);
    }
    return violation;
}

/// @brief runs restarted FISTA without support factorizations, direct gates or orientation pruning
// multiply accepts independent input and output pointers bound to preallocated backend buffers
// profile adds per-call timers in a separate run; its calibrated overhead is reported explicitly
template <typename Multiply>
ReplayResult run_fista(ReplayWorkspace& work, Multiply&& multiply, int limit = 25000, bool profile = false) {
    fdapde_strong_assert(limit > 0, std::invalid_argument, "replay iteration limit must be positive");
    using Clock = std::chrono::steady_clock;
    const auto nanoseconds = [](auto duration) { return std::chrono::duration<double, std::nano>(duration).count(); };
    ReplayResult result;
    double empty_pair_ns = 0.0;
    if (profile) {
        for (int sample = 0; sample < 1024; ++sample) {
            const auto before = Clock::now();
            const auto after = Clock::now();
            empty_pair_ns += nanoseconds(after - before);
        }
        empty_pair_ns /= 1024.0;
    }
    const auto product = [&](const std::vector<double>& input, std::vector<double>& output) {
        if (profile) {
            const auto before = Clock::now();
            multiply(input.data(), output.data());
            result.raw_spmv_ns += nanoseconds(Clock::now() - before);
        } else {
            multiply(input.data(), output.data());
        }
        ++result.spmv_calls;
    };
    const auto check = [&]() {
        product(work.u, work.gradient);
        for (int i = 0; i < work.size; ++i) work.gradient[i] -= work.scaled_c[i];
        return replay_kkt(work) <= work.tolerance;
    };
    const auto start = Clock::now();
    std::copy(work.initial_u.begin(), work.initial_u.end(), work.u.begin());
    std::copy(work.u.begin(), work.u.end(), work.y.begin());
    double momentum = 1.0;
    bool accepted = check();
    for (int iteration = 0; iteration < limit && !accepted; ++iteration) {
        result.iterations = iteration + 1;
        product(work.y, work.gradient);
        bool finite = true;
        for (int i = 0; i < work.size; ++i) {
            work.gradient[i] -= work.scaled_c[i];
            work.next[i] = std::max(0.0, work.y[i] - work.gradient[i] / work.lipschitz);
            finite = finite && std::isfinite(work.next[i]);
        }
        if (!finite) break;
        double restart = 0.0;
        for (int i = 0; i < work.size; ++i) restart += (work.y[i] - work.next[i]) * (work.next[i] - work.u[i]);
        const double next_momentum = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * momentum * momentum));
        if (restart > 0.0) {
            ++result.restarts;
            std::copy(work.next.begin(), work.next.end(), work.y.begin());
            momentum = 1.0;
        } else {
            const double beta = (momentum - 1.0) / next_momentum;
            for (int i = 0; i < work.size; ++i) work.y[i] = work.next[i] + beta * (work.next[i] - work.u[i]);
            momentum = next_momentum;
        }
        work.u.swap(work.next);
        if (!std::all_of(work.y.begin(), work.y.end(), [](double v) { return std::isfinite(v); })) break;
        if (result.iterations % 20 == 0) accepted = check();
    }
    result.full_ns = nanoseconds(Clock::now() - start);
    if (profile) {
        result.timer_overhead_ns = empty_pair_ns * static_cast<double>(result.spmv_calls);
        result.spmv_ns = std::clamp(result.raw_spmv_ns - result.timer_overhead_ns, 0.0, result.full_ns);
    }
    // independently certify the final state and normalize physical weights outside the measured iteration time
    std::fill(work.gradient.begin(), work.gradient.end(), 0.0);
    for (const auto& entry : work.scaled_triplets) work.gradient[entry.row] += entry.value * work.u[entry.col];
    for (int i = 0; i < work.size; ++i) work.gradient[i] -= work.scaled_c[i];
    result.kkt_relative = replay_kkt(work) / work.positive_c_max;
    const double scale = *std::max_element(work.u.begin(), work.u.end());
    result.nonnegative =
      std::all_of(work.u.begin(), work.u.end(), [](double v) { return std::isfinite(v) && v >= 0.0; });
    if (!(scale > 0.0) || !result.nonnegative) {
        result.norm_error = result.weight_relative_error = std::numeric_limits<double>::infinity();
        return result;
    }
    for (int i = 0; i < work.size; ++i) work.weight[i] = work.inverse_diagonal[i] * (work.u[i] / scale);
    const double norm2 = replay_quadratic_form(work.omega, work.weight);
    if (!(norm2 > 0.0) || !std::isfinite(norm2)) {
        result.norm_error = result.weight_relative_error = std::numeric_limits<double>::infinity();
        return result;
    }
    const double norm = std::sqrt(norm2);
    double difference = 0.0, expected_norm2 = 0.0;
    for (int i = 0; i < work.size; ++i) {
        work.weight[i] /= norm;
        const double delta = work.weight[i] - work.expected_weight[i];
        difference += delta * delta;
        expected_norm2 += work.expected_weight[i] * work.expected_weight[i];
        if (work.u[i] > 0.0) {
            ++result.support_count;
            const std::uint32_t index = static_cast<std::uint32_t>(i);
            for (int byte = 0; byte < 4; ++byte) {
                result.support_hash ^= (index >> (8 * byte)) & 0xffU;
                result.support_hash *= 1099511628211ULL;
            }
        }
    }
    result.norm_error = std::abs(replay_quadratic_form(work.omega, work.weight) - 1.0);
    result.weight_relative_error = std::sqrt(difference / expected_norm2);
    result.converged = result.kkt_relative <= 1e-8 && result.norm_error <= 1e-8;
    return result;
}

}   // namespace fdapde_bench

#endif
