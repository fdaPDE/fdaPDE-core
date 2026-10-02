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

#include <fdaPDE/sparse_linear_algebra.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "preallocated_replay.h"

namespace allocation_probe {
thread_local bool active = false;
thread_local std::uint64_t count = 0;
}   // namespace allocation_probe

// count native new calls outside timed rounds; Eigen's malloc-based storage is not measured by this probe
[[gnu::noinline]] void* operator new(std::size_t size) {
    if (void* data = std::malloc(size ? size : 1)) {
        if (allocation_probe::active) ++allocation_probe::count;
        return data;
    }
    throw std::bad_alloc();
}
[[gnu::noinline]] void operator delete(void* data) noexcept { std::free(data); }
[[gnu::noinline]] void operator delete(void* data, std::size_t) noexcept { std::free(data); }

namespace {
using Clock = std::chrono::steady_clock;
using NativeView = fdapde::MatrixView<double, fdapde::Dynamic, 1, fdapde::ColMajor>;
using ConstView = fdapde::MatrixView<const double, fdapde::Dynamic, 1, fdapde::ColMajor>;
using DenseView = fdapde::MatrixView<const double, fdapde::Dynamic, fdapde::Dynamic, fdapde::ColMajor>;
using DenseOwner = fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic, fdapde::ColMajor>;
using SparseCol = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using SparseRow = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
volatile double observation = 0;
const void* volatile constructed_object = nullptr;

/// @brief exposes constructed storage to a standard compiler fence before destruction without traversing it
[[gnu::noinline]] void observe_constructed(const auto& object) {
    constructed_object = &object;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

/// @brief specifies one public workload and the identical process batch used by the comparison runner
struct Options {
    std::string operation = "spmv", backend = "native", api = "preallocated", input;
    int rows = 81, cols = 81, nnz = 19, offset = 0, repetitions = 1, rounds = 3, seed = 20261002;
};

/// @brief rejects malformed counts before allocating operands
int positive(const std::string& value, bool zero = false) {
    std::size_t end = 0;
    const long long count = std::stoll(value, &end);
    fdapde_strong_assert(
      !(end != value.size() || count < (zero ? 0 : 1) || count > std::numeric_limits<int>::max()),
      std::invalid_argument, "invalid integer option");
    return static_cast<int>(count);
}

/// @brief parses a small fixed CLI without accepting unknown options or unconstrained dense shapes
Options parse(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        fdapde_strong_assert(index + 1 < argc, std::invalid_argument, "each option requires a value");
        const std::string key(argv[index]), value(argv[index + 1]);
        if (key == "--op")
            result.operation = value;
        else if (key == "--backend")
            result.backend = value;
        else if (key == "--api")
            result.api = value;
        else if (key == "--input")
            result.input = value;
        else if (key == "--rows")
            result.rows = positive(value);
        else if (key == "--cols")
            result.cols = positive(value);
        else if (key == "--nnz-per-row")
            result.nnz = positive(value);
        else if (key == "--offset")
            result.offset = positive(value, true);
        else if (key == "--repetitions")
            result.repetitions = positive(value);
        else if (key == "--rounds")
            result.rounds = positive(value);
        else if (key == "--seed")
            result.seed = positive(value);
        else
            fdapde_strong_assert(false, std::invalid_argument, "unknown option: " + key);
    }
    fdapde_strong_assert(
      !(result.backend != "native" && result.backend != "eigen-col" && result.backend != "eigen-row"),
      std::invalid_argument, "unsupported backend");
    fdapde_strong_assert(
      !(result.api != "public" && result.api != "preallocated" && result.api != "fused"), std::invalid_argument,
      "unsupported API");
    fdapde_strong_assert(
      result.rounds <= 101 && result.offset <= 1 && result.repetitions <= 10000000, std::invalid_argument,
      "round, offset or batch limit exceeded");
    fdapde_strong_assert(
      !(result.operation != "spmv" && result.operation != "gemv" && result.operation != "gemvt" &&
        result.operation != "assemble" && result.operation != "dense_construct" && result.operation != "project" &&
        result.operation != "momentum" && result.operation != "scale" && result.operation != "replay"),
      std::invalid_argument, "unsupported workload");
    fdapde_strong_assert(
      !((result.operation == "replay") != !result.input.empty() && result.operation != "spmv" &&
        result.operation != "assemble"),
      std::invalid_argument, "input capsules require a sparse workload");
    fdapde_strong_assert(
      !(result.operation == "replay" && result.api != "preallocated"), std::invalid_argument,
      "replay uses preallocated products");
    fdapde_strong_assert(
      result.operation != "replay" || result.repetitions == 1, std::invalid_argument,
      "replay requires one complete call per round");
    fdapde_strong_assert(
      !(result.api == "fused" && result.operation != "project" && result.operation != "momentum" &&
        result.operation != "scale"),
      std::invalid_argument, "disjoint fused assignment requires a coefficient-wise workload");
    fdapde_strong_assert(
      !((result.operation == "gemv" || result.operation == "gemvt" || result.operation == "dense_construct") &&
        std::uint64_t(result.rows) * result.cols > std::numeric_limits<int>::max()),
      std::invalid_argument, "shape exceeds the native coefficient range");
    return result;
}

/// @brief emits valid JSON strings for paths and diagnostics
void json_string(std::string_view value) {
    std::cout << '"';
    for (unsigned char character : value) {
        if (character == '\\' || character == '"')
            std::cout << '\\' << character;
        else if (character < 32)
            std::cout << "\\u00" << "0123456789abcdef"[character / 16] << "0123456789abcdef"[character % 16];
        else
            std::cout << character;
    }
    std::cout << '"';
}

/// @brief generates bounded binary fractions identically for independently owned native and Eigen inputs
double value(int row, int column, int seed) {
    return ((7 * (row % 17) + 3 * (column % 17) + 5 * (seed % 17)) % 17 - 8) / 8.0;
}

/// @brief hashes canonical input bytes without deriving native storage from Eigen
std::uint64_t hash_value(std::uint64_t hash, double coefficient) {
    return (hash ^ std::bit_cast<std::uint64_t>(coefficient)) * 1099511628211ULL;
}

/// @brief times complete calls after an untimed warmup and retains every round
std::vector<double> measure(const Options& options, const std::function<void()>& operation) {
    std::vector<double> samples(options.rounds);
    operation();
    for (double& sample : samples) {
        const auto start = Clock::now();
        for (int repetition = 0; repetition < options.repetitions; ++repetition) operation();
        sample = std::chrono::duration<double, std::nano>(Clock::now() - start).count() / options.repetitions;
    }
    return samples;
}

/// @brief validates every coefficient against an independent scalar accumulation outside the timed region
double verify(const double* actual, const std::vector<double>& expected, const std::vector<double>& absolute) {
    double maximum = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double difference = std::abs(actual[index] - expected[index]);
        fdapde_strong_assert(
          std::isfinite(actual[index]) && difference <= 1e-11 * std::max(1.0, absolute[index]), std::runtime_error,
          "public coefficient disagrees with the scalar oracle");
        maximum = std::max(maximum, difference);
    }
    return maximum;
}

/// @brief compares dense, sparse and fused public calls with independently prepared operands
int run(const Options& options) {
    Eigen::setNbThreads(1);
    const bool native = options.backend == "native";
    const bool sparse = options.operation == "spmv" || options.operation == "assemble" || options.operation == "replay";
    int rows = options.rows, cols = options.cols;
    std::vector<fdapde_bench::ReplayTriplet> entries;
    fdapde_bench::ReplayInput replay_input;
    fdapde_bench::ReplayWorkspace workspace;
    if (!options.input.empty()) {
        replay_input = fdapde_bench::load_replay(options.input);
        rows = cols = replay_input.size;
        workspace = fdapde_bench::prepare_replay(replay_input);
        entries = workspace.scaled_triplets;
    } else if (sparse) {
        const int width = std::min(options.nnz, cols);
        const int step = std::gcd(cols, 17) == 1 ? 17 : 1;
        fdapde_strong_assert(
          std::uint64_t(rows) * width <= std::numeric_limits<int>::max(), std::invalid_argument,
          "too many sparse entries");
        entries.reserve(std::size_t(rows) * width);
        for (int row = 0; row < rows; ++row) {
            for (int index = 0; index < width; ++index) {
                const int col = static_cast<int>((std::int64_t(row) + std::int64_t(index) * step) % cols);
                entries.push_back({row, col, row == col ? 2.0 : (index % 2 ? -0.03125 : 0.0625)});
            }
        }
    }
    const int input_size = options.operation == "gemvt" ? rows : cols;
    const int output_size = options.operation == "gemvt" ? cols : rows;
    std::vector<double> native_input(std::size_t(input_size) + options.offset),
      eigen_input(std::size_t(input_size) + options.offset);
    std::vector<double> native_output(std::size_t(output_size) + options.offset),
      eigen_output(std::size_t(output_size) + options.offset);
    std::vector<double> second(output_size), third(output_size), eigen_second(output_size), eigen_third(output_size);
    std::uint64_t input_hash = 1469598103934665603ULL;
    for (int index = 0; index < input_size; ++index) {
        const double coefficient = options.input.empty() ? value(index, 0, options.seed) : workspace.initial_u[index];
        native_input[index + options.offset] = coefficient;
        eigen_input[index + options.offset] = coefficient;
        input_hash = hash_value(input_hash, coefficient);
    }
    for (int index = 0; index < output_size; ++index) {
        second[index] = eigen_second[index] = value(index, 1, options.seed);
        third[index] = eigen_third[index] = value(index, 2, options.seed);
        if (
          !sparse && options.operation != "gemv" && options.operation != "gemvt" &&
          options.operation != "dense_construct") {
            input_hash = hash_value(input_hash, second[index]);
            if (options.operation == "momentum") input_hash = hash_value(input_hash, third[index]);
        }
    }
    ConstView input(native_input.data() + options.offset, input_size);
    NativeView output(native_output.data() + options.offset, output_size);
    ConstView gradient(second.data(), output_size), old(third.data(), output_size);
    Eigen::Map<const Eigen::VectorXd> eigen_x(eigen_input.data() + options.offset, input_size);
    Eigen::Map<Eigen::VectorXd> eigen_y(eigen_output.data() + options.offset, output_size);
    Eigen::Map<const Eigen::VectorXd> eigen_g(eigen_second.data(), output_size),
      eigen_u(eigen_third.data(), output_size);
    if (options.operation == "replay") {
        for (double coefficient : workspace.scaled_c) input_hash = hash_value(input_hash, coefficient);
        for (const auto& entry : workspace.omega) input_hash = hash_value(input_hash, entry.value);
        for (double coefficient : workspace.expected_weight) input_hash = hash_value(input_hash, coefficient);
    }
    std::vector<double> native_dense, eigen_dense;
    std::vector<fdapde::Triplet<double>> native_triplets;
    std::vector<Eigen::Triplet<double>> eigen_triplets;
    fdapde::SparseMatrix<double> native_sparse;
    SparseCol eigen_col;
    SparseRow eigen_row;
    if (sparse) {
        native_triplets.reserve(entries.size());
        eigen_triplets.reserve(entries.size());
        for (const auto& entry : entries) {
            native_triplets.emplace_back(entry.row, entry.col, entry.value);
            eigen_triplets.emplace_back(entry.row, entry.col, entry.value);
            input_hash = hash_value(input_hash, entry.row);
            input_hash = hash_value(input_hash, entry.col);
            input_hash = hash_value(input_hash, entry.value);
        }
        native_sparse = fdapde::SparseMatrix<double>(rows, cols, native_triplets);
        eigen_col.resize(rows, cols);
        eigen_col.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
        eigen_row.resize(rows, cols);
        eigen_row.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
    } else if (options.operation == "gemv" || options.operation == "gemvt" || options.operation == "dense_construct") {
        native_dense.resize(std::size_t(rows) * cols + options.offset);
        eigen_dense.resize(std::size_t(rows) * cols + options.offset);
        for (int column = 0; column < cols; ++column)
            for (int row = 0; row < rows; ++row) {
                const double coefficient = value(row, column, options.seed);
                native_dense[std::size_t(column) * rows + row + options.offset] = coefficient;
                eigen_dense[std::size_t(column) * rows + row + options.offset] = coefficient;
                input_hash = hash_value(input_hash, coefficient);
            }
    } else
        fdapde_strong_assert(
          input_size == output_size, std::invalid_argument, "coefficient-wise operands require equal sizes");
    DenseView matrix(
      native_dense.data() + (native_dense.empty() ? 0 : options.offset), rows, native_dense.empty() ? 0 : cols);
    Eigen::Map<const Eigen::MatrixXd> eigen_matrix(
      eigen_dense.data() + (eigen_dense.empty() ? 0 : options.offset), rows, eigen_dense.empty() ? 0 : cols);
    std::function<void()> operation;
    fdapde_bench::ReplayResult replay_result, profiled;
    const auto sparse_product = [&](const double* x, double* y) {
        if (native) {
            ConstView source(x, cols);
            NativeView destination(y, rows);
            native_sparse.multiply_into(source, destination);
        } else {
            Eigen::Map<const Eigen::VectorXd> source(x, cols);
            Eigen::Map<Eigen::VectorXd> destination(y, rows);
            if (options.backend == "eigen-row")
                destination.noalias() = eigen_row * source;
            else
                destination.noalias() = eigen_col * source;
        }
    };
    if (options.operation == "replay") {
        operation = [&] { replay_result = fdapde_bench::run_fista(workspace, sparse_product, 25000, false); };
    } else if (options.operation == "assemble") {
        operation = [&] {
            if (native) {
                fdapde::SparseMatrix<double> result(rows, cols, native_triplets);
                observe_constructed(result);
                observation = result.non_zeros();
            } else if (options.backend == "eigen-row") {
                SparseRow result(rows, cols);
                result.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
                observe_constructed(result);
                observation = result.nonZeros();
            } else {
                SparseCol result(rows, cols);
                result.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
                observe_constructed(result);
                observation = result.nonZeros();
            }
        };
    } else if (options.operation == "dense_construct") {
        operation = [&] {
            if (native) {
                DenseOwner result(matrix);
                observe_constructed(result);
                observation = result.data()[0];
            } else {
                Eigen::MatrixXd result(eigen_matrix);
                observe_constructed(result);
                observation = result.data()[0];
            }
        };
    } else if (options.operation == "spmv") {
        operation = [&] {
            if (options.api == "public") {
                if (native) {
                    output = native_sparse * input;
                } else if (options.backend == "eigen-row") {
                    const Eigen::VectorXd temporary = eigen_row * eigen_x;
                    eigen_y = temporary;
                } else {
                    const Eigen::VectorXd temporary = eigen_col * eigen_x;
                    eigen_y = temporary;
                }
            } else
                sparse_product(native ? input.data() : eigen_x.data(), native ? output.data() : eigen_y.data());
        };
    } else if (options.operation == "gemv" || options.operation == "gemvt") {
        operation = [&] {
            if (native) {
                if (options.operation == "gemvt") {
                    if (options.api == "public")
                        output = matrix.transpose() * input;
                    else
                        matrix.transpose().multiply_into(input, output);
                } else {
                    if (options.api == "public")
                        output = matrix * input;
                    else
                        matrix.multiply_into(input, output);
                }
            } else {
                if (options.api == "public") {
                    const Eigen::VectorXd temporary = options.operation == "gemvt" ?
                                                        Eigen::VectorXd(eigen_matrix.transpose() * eigen_x) :
                                                        Eigen::VectorXd(eigen_matrix * eigen_x);
                    eigen_y = temporary;
                } else if (options.operation == "gemvt")
                    eigen_y.noalias() = eigen_matrix.transpose() * eigen_x;
                else
                    eigen_y.noalias() = eigen_matrix * eigen_x;
            }
        };
    } else {
        operation = [&] {
            if (!native) {
                if (options.operation == "project")
                    eigen_y = (eigen_x - eigen_g / 2.0).cwiseMax(0.0);
                else if (options.operation == "momentum")
                    eigen_y = eigen_x + 0.125 * (eigen_x - eigen_u);
                else
                    eigen_y.array() = eigen_g.array() * eigen_x.array();
            } else if (options.api == "fused") {
                if (options.operation == "project")
                    output.assign_disjoint((input - gradient / 2.0).cwise().apply([](double coefficient) {
                        return std::max(0.0, coefficient);
                    }));
                else if (options.operation == "momentum")
                    output.assign_disjoint(input + 0.125 * (input - old));
                else
                    output.assign_disjoint(gradient.cwise() * input.cwise());
            } else {
                if (options.operation == "project")
                    output = (input - gradient / 2.0).cwise().apply([](double coefficient) {
                        return std::max(0.0, coefficient);
                    });
                else if (options.operation == "momentum")
                    output = input + 0.125 * (input - old);
                else
                    output = gradient.cwise() * input.cwise();
            }
        };
    }
    std::vector<double> expected(output_size), absolute(output_size);
    if (options.operation == "spmv" || options.operation == "assemble") {
        for (const auto& entry : entries) {
            expected[entry.row] += entry.value * input.data()[entry.col];
            absolute[entry.row] += std::abs(entry.value * input.data()[entry.col]);
        }
    } else if (options.operation == "gemv" || options.operation == "gemvt") {
        for (int row = 0; row < output_size; ++row)
            for (int inner = 0; inner < input_size; ++inner) {
                const double coefficient = options.operation == "gemvt" ? matrix(inner, row) : matrix(row, inner);
                expected[row] += coefficient * input.data()[inner];
                absolute[row] += std::abs(coefficient * input.data()[inner]);
            }
    } else if (options.operation != "replay" && options.operation != "dense_construct")
        for (int index = 0; index < output_size; ++index) {
            if (options.operation == "project")
                expected[index] = std::max(0.0, input.data()[index] - gradient.data()[index] / 2.0);
            else if (options.operation == "momentum")
                expected[index] = input.data()[index] + 0.125 * (input.data()[index] - old.data()[index]);
            else
                expected[index] = gradient.data()[index] * input.data()[index];
            absolute[index] = std::abs(expected[index]);
        }
    operation();
    if (options.operation == "dense_construct") {
        const DenseOwner copied(matrix);
        const Eigen::MatrixXd eigen_copied(eigen_matrix);
        for (int column = 0; column < cols; ++column)
            for (int row = 0; row < rows; ++row) {
                fdapde_strong_assert(
                  !(copied(row, column) != value(row, column, options.seed) ||
                    eigen_copied(row, column) != value(row, column, options.seed)),
                  std::runtime_error, "dense construction failed the canonical coefficient oracle");
            }
    }
    // check the canonical sparse coefficients and structure before timing assembly calls
    if (options.operation == "assemble") {
        fdapde_strong_assert(
          native_sparse.non_zeros() == static_cast<int>(entries.size()) &&
            eigen_col.nonZeros() == native_sparse.non_zeros() && eigen_row.nonZeros() == native_sparse.non_zeros(),
          std::runtime_error, "assembly changed the canonical nonzero structure");
        for (const auto& entry : entries) {
            fdapde_strong_assert(
              native_sparse.coeff(entry.row, entry.col) == entry.value &&
                eigen_col.coeff(entry.row, entry.col) == entry.value &&
                eigen_row.coeff(entry.row, entry.col) == entry.value,
              std::runtime_error, "assembly changed a canonical coefficient");
        }
        sparse_product(native ? input.data() : eigen_x.data(), native ? output.data() : eigen_y.data());
        verify(native ? output.data() : eigen_y.data(), expected, absolute);
    }
    double error = 0;
    const bool output_check =
      options.operation != "assemble" && options.operation != "dense_construct" && options.operation != "replay";
    if (output_check) error = verify(native ? output.data() : eigen_y.data(), expected, absolute);
    allocation_probe::count = 0;
    allocation_probe::active = native;
    try {
        operation();
    } catch (...) {
        allocation_probe::active = false;
        throw;
    }
    allocation_probe::active = false;
    const auto allocations = allocation_probe::count;
    fdapde_strong_assert(
      !(native && options.api == "preallocated" &&
        (options.operation == "spmv" || options.operation == "gemv" || options.operation == "gemvt") && allocations),
      std::runtime_error, "disjoint preallocated product allocated storage");
    fdapde_strong_assert(
      !native || options.api != "fused" || allocations == 0, std::runtime_error,
      "disjoint fused assignment allocated storage");
    std::vector<double> fista_samples;
    const std::function<void()> timed_operation = [&] {
        operation();
        if (options.operation == "replay") fista_samples.push_back(replay_result.full_ns);
    };
    // replay median_ns covers the complete harness call, including independent final verification
    // fista_timings_ns separately reports the uninstrumented initialization and iteration loop
    if (options.operation == "replay") fista_samples.reserve(std::size_t(options.rounds) * options.repetitions + 1);
    const auto samples = measure(options, options.operation == "replay" ? timed_operation : operation);
    if (!fista_samples.empty()) fista_samples.erase(fista_samples.begin());
    if (output_check) error = verify(native ? output.data() : eigen_y.data(), expected, absolute);
    if (options.operation == "replay") {
        profiled = fdapde_bench::run_fista(workspace, sparse_product, 25000, true);
        fdapde_strong_assert(
          profiled.converged && profiled.nonnegative && profiled.kkt_relative <= 1.0001e-8 &&
            profiled.norm_error <= 1e-8 && std::isfinite(profiled.weight_relative_error) &&
            profiled.iterations == replay_result.iterations && profiled.restarts == replay_result.restarts &&
            profiled.support_hash == replay_result.support_hash &&
            profiled.support_count == replay_result.support_count,
          std::runtime_error, "profiled replay differs from the certified uninstrumented trajectory");
    }
    auto ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    double checksum = 0;
    if (output_check)
        for (int index = 0; index < output_size; ++index)
            checksum += (native ? output.data()[index] : eigen_y.data()[index]) * (1 + index % 11);
    fdapde_strong_assert(
      !(options.operation == "replay" && (!replay_result.converged || !replay_result.nonnegative ||
                                          replay_result.kkt_relative > 1.0001e-8 || replay_result.norm_error > 1e-8)),
      std::runtime_error, "FISTA-only replay failed its independent final certificate");
    std::cout << std::setprecision(17) << "{\"verified\":true,\"op\":";
    json_string(options.operation);
    std::cout << ",\"oracle\":";
    json_string(
      options.operation == "replay"                                             ? "independent-kkt" :
      options.operation == "assemble" || options.operation == "dense_construct" ? "canonical-construction" :
                                                                                  "scalar-coefficients");
    std::cout << ",\"backend\":";
    json_string(options.backend);
    std::cout << ",\"api\":";
    json_string(options.api);
    std::cout << ",\"input\":";
    json_string(options.input);
    std::cout << ",\"rows\":" << rows << ",\"cols\":" << cols << ",\"nnz\":" << native_sparse.non_zeros()
              << ",\"offset\":" << options.offset << ",\"seed\":" << options.seed << ",\"input_hash\":\"" << std::hex
              << input_hash << std::dec << "\",\"assignment\":" << FDAPDE_ENABLE_SIMD_ASSIGNMENT
              << ",\"product\":" << FDAPDE_ENABLE_SIMD_PRODUCT << ",\"eigen_version\":\"" << EIGEN_WORLD_VERSION << '.'
              << EIGEN_MAJOR_VERSION << '.' << EIGEN_MINOR_VERSION << "\",\"internal_threads\":" << Eigen::nbThreads()
              << ",\"repetitions\":" << options.repetitions << ",\"rounds\":" << options.rounds
              << ",\"new_allocations_per_call\":";
    if (native)
        std::cout << allocations;
    else
        std::cout << "null";
    std::cout << ",\"max_abs_error\":" << error << ",\"checksum\":" << checksum << ",\"working_set_bytes\":"
              << (sparse ? entries.size() * 12 + (std::size_t(rows) + 1) * 4 : native_dense.size() * 8) +
                   (std::size_t(input_size) + output_size) * 8 +
                   (options.operation == "replay" ? std::size_t(rows) * 64 :
                    (options.operation == "project" || options.operation == "momentum" ||
                     options.operation == "scale") ?
                                                    std::size_t(rows) * 8 :
                                                    0)
              << ",\"median_ns\":" << ordered[ordered.size() / 2] << ",\"timings_ns\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) std::cout << (index ? "," : "") << samples[index];
    if (options.operation == "replay") {
        std::cout << "],\"replay\":{\"full_ns\":" << replay_result.full_ns
                  << ",\"profile_full_ns\":" << profiled.full_ns << ",\"profile_spmv_ns\":" << profiled.spmv_ns
                  << ",\"raw_spmv_ns\":" << profiled.raw_spmv_ns
                  << ",\"timer_overhead_ns\":" << profiled.timer_overhead_ns
                  << ",\"profile_iterations\":" << profiled.iterations << ",\"profile_restarts\":" << profiled.restarts
                  << ",\"spmv_calls\":" << profiled.spmv_calls << ",\"iterations\":" << replay_result.iterations
                  << ",\"restarts\":" << replay_result.restarts << ",\"support_count\":" << replay_result.support_count
                  << ",\"support_hash\":\"" << std::hex << replay_result.support_hash << std::dec
                  << "\",\"kkt_relative\":" << replay_result.kkt_relative
                  << ",\"norm_error\":" << replay_result.norm_error
                  << ",\"weight_relative_error\":" << replay_result.weight_relative_error
                  << ",\"nonnegative\":true,\"converged\":true,\"fista_timings_ns\":[";
        for (std::size_t index = 0; index < fista_samples.size(); ++index)
            std::cout << (index ? "," : "") << fista_samples[index];
        std::cout << "],\"final_weight\":[";
        for (int index = 0; index < rows; ++index) std::cout << (index ? "," : "") << workspace.weight[index];
        std::cout << "]}";
    } else
        std::cout << ']';
    std::cout << ",\"error\":null}\n";
    return 0;
}
}   // namespace

int main(int argc, char** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& error) {
        allocation_probe::active = false;
        std::cout << "{\"verified\":false,\"error\":";
        json_string(error.what());
        std::cout << "}\n";
        return 1;
    }
}
