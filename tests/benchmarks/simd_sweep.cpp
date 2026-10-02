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
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
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

/// @brief stores one explicitly selected benchmark case and its public-call repetition schedule
struct Options {
    std::string_view suite, name;
    int size = 0, repetitions = 1, rounds = 5;
};

/// @brief distinguishes the five public assignment operations without branching inside coefficient loops
enum class Assignment {
    Scale,
    Copy,
    Broadcast,
    Add,
    Affine
};

/// @brief parses a positive integer without accepting trailing characters or overflowing its representation
int positive_integer(std::string_view value) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    fdapde_strong_assert(
      parsed.ec == std::errc {} && parsed.ptr == value.data() + value.size() && result > 0, std::invalid_argument,
      "numeric arguments must be positive integers");
    return result;
}

/// @brief rejects incomplete or unsupported command-line options before allocating benchmark storage
Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i += 2) {
        fdapde_strong_assert(i + 1 < argc, std::invalid_argument, "every option requires a value");
        const std::string_view key(argv[i]), value(argv[i + 1]);
        if (key == "--suite")
            options.suite = value;
        else if (key == "--case")
            options.name = value;
        else if (key == "--size")
            options.size = positive_integer(value);
        else if (key == "--repetitions")
            options.repetitions = positive_integer(value);
        else if (key == "--rounds")
            options.rounds = positive_integer(value);
        else
            throw std::invalid_argument("unknown benchmark option");
    }
    fdapde_strong_assert(
      (options.suite == "assignment" || options.suite == "product") && !options.name.empty() && options.size > 0,
      std::invalid_argument, "require --suite assignment|product --case NAME --size N");
    fdapde_strong_assert(
      options.repetitions <= 10000000 && options.rounds <= 101, std::invalid_argument,
      "repetitions exceed 10000000 or rounds exceed 101");
    return options;
}

/// @brief validates runtime shapes independently of the core debug-assertion mode
int coefficient_count(int rows, int cols) {
    const std::uint64_t size = std::uint64_t(rows) * std::uint64_t(cols);
    fdapde_strong_assert(
      rows > 0 && cols > 0 && size <= std::uint64_t(std::numeric_limits<int>::max()), std::invalid_argument,
      "benchmark matrix shape exceeds the supported coefficient range");
    return static_cast<int>(size);
}

/// @brief computes bounded, nonconstant binary-fraction inputs from logical coordinates
double input_value(int i, int j, int seed) { return ((7 * (i % 17) + 3 * (j % 17) + 5 * seed) % 17 - 8) / 4.0; }

/// @brief initializes each operand using logical coordinates independently of its storage order
template <typename Matrix> void initialize(Matrix& matrix, int seed) {
    using Scalar = typename Matrix::Scalar;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) matrix(i, j) = Scalar(input_value(i, j, seed));
    }
}

/// @brief emits one escaped JSON string, including arbitrary diagnostic text from caught exceptions
void json_string(std::string_view value) {
    std::putchar('"');
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\')
            std::printf("\\%c", ch);
        else if (ch < 0x20)
            std::printf("\\u%04x", ch);
        else
            std::putchar(ch);
    }
    std::putchar('"');
}

/// @brief times only repeated public calls after one untimed warmup, retaining every round for the runner
template <typename Operation> std::vector<double> measure(const Options& options, Operation operation) {
    std::vector<double> samples(options.rounds);
    std::uint64_t call = 0;
    operation(call++);
    for (double& sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < options.repetitions; ++repeat) operation(call++);
        sample = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() /
                 options.repetitions;
    }
    return samples;
}

/// @brief reports exact shape, build switches, samples and release-active coefficient verification in one JSON row
void report(
  const Options& options, int rows, int inner, int cols, const char* scalar, int lhs_order, int rhs_order,
  int output_order, std::uint64_t buffer_bytes, std::uint64_t working_set_bytes, std::vector<double> samples,
  double checksum) {
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2;
    const double median = sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    std::printf("{\"suite\":");
    json_string(options.suite);
    std::printf(",\"case\":");
    json_string(options.name);
    std::printf(
      ",\"size\":%d,\"rows\":%d,\"inner\":%d,\"cols\":%d,\"coefficients\":%d,\"scalar\":\"%s\","
      "\"lhs_order\":%d,\"rhs_order\":%d,\"output_order\":%d,\"repetitions\":%d,\"rounds\":%d,"
      "\"assignment\":%d,\"product\":%d,\"assertions_enabled\":",
      options.size, rows, inner, cols, coefficient_count(rows, cols), scalar, lhs_order, rhs_order, output_order,
      options.repetitions, options.rounds, FDAPDE_ENABLE_SIMD_ASSIGNMENT, FDAPDE_ENABLE_SIMD_PRODUCT);
#ifdef FDAPDE_NO_DEBUG
    std::printf("false");
#else
    std::printf("true");
#endif
    std::printf(",\"fast_math\":");
#ifdef __FAST_MATH__
    std::printf("true");
#else
    std::printf("false");
#endif
    std::printf(",\"compiler\":");
#if defined(__clang__)
    json_string(__clang_version__);
#elif defined(__GNUC__)
    json_string(__VERSION__);
#else
    json_string("unknown");
#endif
    std::printf(
      ",\"buffer_bytes\":%llu,\"working_set_bytes\":%llu,\"timings_ns\":[",
      static_cast<unsigned long long>(buffer_bytes), static_cast<unsigned long long>(working_set_bytes));
    for (std::size_t i = 0; i < samples.size(); ++i) std::printf("%s%.17g", i ? "," : "", samples[i]);
    std::printf(
      "],\"median_ns\":%.17g,\"checksum\":%.17g,\"calls\":%llu,\"verified\":true,\"warnings\":[", median, checksum,
      static_cast<unsigned long long>(1 + std::uint64_t(options.repetitions) * options.rounds));
    const bool short_round = median * options.repetitions < 1000000;
    if (short_round) json_string("round duration below 1 ms; increase repetitions");
    if (options.name.starts_with("fem") && options.size != rows) {
        if (short_round) std::putchar(',');
        json_string("fixed FEM case ignores --size; use reported shape");
    }
    std::printf("],\"error\":null}\n");
}

/// @brief executes a public assignment in a separate call while retaining its aliasing and snapshot semantics
template <Assignment Op, typename Output, typename Lhs, typename Rhs>
FDAPDE_BENCH_NOINLINE void assign(
  Output& output, [[maybe_unused]] const Lhs& lhs, [[maybe_unused]] const Rhs& rhs,
  [[maybe_unused]] typename Output::Scalar scalar) {
    using Scalar = typename Output::Scalar;
    if constexpr (Op == Assignment::Scale)
        output *= scalar;
    else if constexpr (Op == Assignment::Copy)
        output = lhs;
    else if constexpr (Op == Assignment::Broadcast)
        output.cwise() = scalar;
    else if constexpr (Op == Assignment::Add)
        output += lhs;
    else
        output = Scalar(0.5) * lhs + Scalar(0.25) * rhs;
}

/// @brief checks every assignment coefficient against its original binary inputs and the exact public-call count
template <Assignment Op, typename Output, typename Lhs, typename Rhs>
void assignment_case(const Options& options, Output& output, Lhs& lhs, Rhs& rhs) {
    using Scalar = typename Output::Scalar;
    initialize(lhs, 0);
    initialize(rhs, 1);
    initialize(output, 2);
    const auto samples = measure(options, [&](std::uint64_t call) {
        const double scalar = Op == Assignment::Broadcast ? (call % 2 ? -1.75 : 3.25) : (call % 2 ? 0.5 : 2.0);
        assign<Op>(output, lhs, rhs, Scalar(scalar));
    });
    const std::uint64_t calls = 1 + std::uint64_t(options.repetitions) * options.rounds;
    double checksum = 0;
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            double expected = 0;
            if constexpr (Op == Assignment::Scale)
                expected = input_value(i, j, 2) * (calls % 2 ? 2.0 : 1.0);
            else if constexpr (Op == Assignment::Copy)
                expected = lhs.rows() == output.rows() && lhs.cols() == output.cols() ?
                             input_value(i, j, 0) :
                             (lhs.rows() == 1 ? input_value(0, i + j, 0) : input_value(i + j, 0, 0));
            else if constexpr (Op == Assignment::Broadcast)
                expected = calls % 2 ? 3.25 : -1.75;
            else if constexpr (Op == Assignment::Add)
                expected = input_value(i, j, 2) + double(calls) * input_value(i, j, 0);
            else
                expected = 0.5 * input_value(i, j, 0) + 0.25 * input_value(i, j, 1);
            // verify each logical coefficient against the exact binary-input oracle even in release builds
            fdapde_strong_assert(
              output(i, j) == expected, std::runtime_error, "assignment coefficient differs from scalar oracle");
            checksum += output(i, j);
        }
    }
    report(
      options, output.rows(), 0, output.cols(), std::is_same_v<Scalar, float> ? "float" : "double", Lhs::StorageOrder,
      Rhs::StorageOrder, Output::StorageOrder, std::uint64_t(output.rows()) * output.cols() * sizeof(Scalar),
      (std::uint64_t(output.rows()) * output.cols() + std::uint64_t(lhs.rows()) * lhs.cols() +
       std::uint64_t(rhs.rows()) * rhs.cols()) *
        sizeof(Scalar),
      samples, checksum);
}

/// @brief selects one assignment operation without introducing a runtime branch in its timed kernel
template <typename Matrix>
void assignment_operation(const Options& options, std::string_view name, Matrix output, Matrix lhs, Matrix rhs) {
    if (name == "scale")
        assignment_case<Assignment::Scale>(options, output, lhs, rhs);
    else if (name == "copy")
        assignment_case<Assignment::Copy>(options, output, lhs, rhs);
    else if (name == "broadcast")
        assignment_case<Assignment::Broadcast>(options, output, lhs, rhs);
    else if (name == "add")
        assignment_case<Assignment::Add>(options, output, lhs, rhs);
    else if (name == "affine")
        assignment_case<Assignment::Affine>(options, output, lhs, rhs);
    else
        throw std::invalid_argument("unknown assignment operation");
}

/// @brief chooses a scalar-aligned pointer that deliberately misses sixteen-byte SIMD alignment
double* unaligned_data(std::vector<double>& storage) {
    return storage.data() + (reinterpret_cast<std::uintptr_t>(storage.data()) % 16 == 0 ? 1 : 0);
}

/// @brief allocates independent assignment operands for each public vector, short-axis matrix or external view case
void run_assignment(const Options& options) {
    const int n = options.size;
    if (options.name == "vector_float_affine") {
        Vector<float, Dynamic> output(n), lhs(n), rhs(n);
        assignment_case<Assignment::Affine>(options, output, lhs, rhs);
        return;
    }
    if (options.name == "vector_orientation_copy") {
        std::vector<double> output_storage(n), lhs_storage(n), rhs_storage(n);
        VectorView<double, Dynamic> output(output_storage.data(), n);
        MatrixView<double, 1, Dynamic> lhs(lhs_storage.data(), n), rhs(rhs_storage.data(), n);
        assignment_case<Assignment::Copy>(options, output, lhs, rhs);
        return;
    }
    if (options.name == "row3_float_scale" || options.name == "cross_layout_copy") {
        fdapde_strong_assert(
          n % 3 == 0, std::invalid_argument, "short-axis assignment sizes must be divisible by three");
        (void)coefficient_count(n / 3, 3);
        if (options.name == "row3_float_scale") {
            Matrix<float, Dynamic, Dynamic> output(n / 3, 3), lhs(n / 3, 3), rhs(n / 3, 3);
            assignment_case<Assignment::Scale>(options, output, lhs, rhs);
        } else {
            Matrix<double, Dynamic, Dynamic, ColMajor> output(n / 3, 3);
            Matrix<double, Dynamic, Dynamic> lhs(n / 3, 3), rhs(n / 3, 3);
            assignment_case<Assignment::Copy>(options, output, lhs, rhs);
        }
        return;
    }
    const auto separator = options.name.find('_');
    fdapde_strong_assert(
      separator != std::string_view::npos, std::invalid_argument, "assignment case needs shape_operation");
    const auto shape = options.name.substr(0, separator), operation = options.name.substr(separator + 1);
    fdapde_strong_assert(
      !(shape == "row3" || shape == "col3" || shape == "view") || n % 3 == 0, std::invalid_argument,
      "short-axis assignment sizes must be divisible by three");
    if (shape == "vector") {
        assignment_operation(
          options, operation, Vector<double, Dynamic>(n), Vector<double, Dynamic>(n), Vector<double, Dynamic>(n));
    } else if (shape == "row3") {
        (void)coefficient_count(n / 3, 3);
        using Matrix = fdapde::Matrix<double, Dynamic, Dynamic>;
        assignment_operation(options, operation, Matrix(n / 3, 3), Matrix(n / 3, 3), Matrix(n / 3, 3));
    } else if (shape == "col3") {
        (void)coefficient_count(3, n / 3);
        using Matrix = fdapde::Matrix<double, Dynamic, Dynamic, ColMajor>;
        assignment_operation(options, operation, Matrix(3, n / 3), Matrix(3, n / 3), Matrix(3, n / 3));
    } else if (shape == "view") {
        const int count = coefficient_count(n / 3, 3);
        std::vector<double> output(std::size_t(count) + 3), lhs(std::size_t(count) + 3), rhs(std::size_t(count) + 3);
        using View = MatrixView<double, Dynamic, Dynamic>;
        assignment_operation(
          options, operation, View(unaligned_data(output), n / 3, 3), View(unaligned_data(lhs), n / 3, 3),
          View(unaligned_data(rhs), n / 3, 3));
    } else
        throw std::invalid_argument("unknown assignment shape");
}

/// @brief evaluates the public product assignment without removing its snapshot, allocation or output-copy costs
template <typename Result, typename Lhs, typename Rhs>
FDAPDE_BENCH_NOINLINE void product(Result& output, const Lhs& lhs, const Rhs& rhs) {
    output = lhs * rhs;
}

/// @brief constructs a column-major owner directly from a mixed-layout product without a final matrix assignment
template <typename Result, typename Lhs, typename Rhs>
FDAPDE_BENCH_NOINLINE void construct_product(std::optional<Result>& output, const Lhs& lhs, const Rhs& rhs) {
    output.emplace(lhs * rhs);
}

/// @brief verifies every product coefficient using increasing-k scalar accumulation outside the timed calls
template <bool Construct = false, typename Result, typename Lhs, typename Rhs>
void product_case(const Options& options, Result output, Lhs lhs, Rhs rhs) {
    using Scalar = std::common_type_t<typename Lhs::Scalar, typename Rhs::Scalar>;
    initialize(lhs, 0);
    initialize(rhs, 1);
    const int rows = lhs.rows(), inner = lhs.cols(), cols = rhs.cols();
    std::vector<Scalar> expected(coefficient_count(rows, cols), Scalar(0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < inner; ++k) expected[i * cols + j] += lhs(i, k) * rhs(k, j);
        }
    }
    std::optional<Result> constructed;
    const auto samples = measure(options, [&](std::uint64_t) {
        if constexpr (Construct)
            construct_product(constructed, lhs, rhs);
        else
            product(output, lhs, rhs);
    });
    const Result& actual = [&]() -> const Result& {
        if constexpr (Construct)
            return *constructed;
        else
            return output;
    }();
    // verify logical dimensions before checking the complete ordered scalar oracle
    fdapde_strong_assert(
      actual.rows() == rows && actual.cols() == cols, std::runtime_error, "product returned an incorrect shape");
    double checksum = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // verify every output coefficient using the original increasing-k accumulation order
            fdapde_strong_assert(
              actual(i, j) == expected[i * cols + j], std::runtime_error,
              "product coefficient differs from ordered scalar oracle");
            checksum += actual(i, j);
        }
    }
    report(
      options, rows, inner, cols, std::is_same_v<Scalar, float> ? "float" : "double", Lhs::StorageOrder,
      Rhs::StorageOrder, Result::StorageOrder, std::uint64_t(rows) * cols * sizeof(Scalar),
      (std::uint64_t(rows) * cols + std::uint64_t(rows) * inner + std::uint64_t(inner) * cols) * sizeof(Scalar),
      samples, checksum);
}

/// @brief allocates dynamic dense product operands after validating all three runtime matrix shapes
template <typename Scalar, int LhsOrder, int RhsOrder, int OutputOrder>
void dynamic_product(const Options& options, int rows, int inner, int cols) {
    (void)coefficient_count(rows, inner);
    (void)coefficient_count(inner, cols);
    (void)coefficient_count(rows, cols);
    product_case(
      options, Matrix<Scalar, Dynamic, Dynamic, OutputOrder>(rows, cols),
      Matrix<Scalar, Dynamic, Dynamic, LhsOrder>(rows, inner), Matrix<Scalar, Dynamic, Dynamic, RhsOrder>(inner, cols));
}

/// @brief selects square, mixed-layout, odd, rectangular, unaligned or fixed fem-sized public product cases
void run_product(const Options& options) {
    const int n = options.size;
    if (options.name == "square_row")
        dynamic_product<double, RowMajor, RowMajor, RowMajor>(options, n, n, n);
    else if (options.name == "square_col")
        dynamic_product<double, ColMajor, ColMajor, ColMajor>(options, n, n, n);
    else if (options.name == "square_float")
        dynamic_product<float, RowMajor, RowMajor, RowMajor>(options, n, n, n);
    else if (options.name == "square_float_col")
        dynamic_product<float, ColMajor, ColMajor, ColMajor>(options, n, n, n);
    else if (options.name == "mixed_row")
        dynamic_product<double, RowMajor, ColMajor, RowMajor>(options, n, n, n);
    else if (options.name == "mixed_col")
        dynamic_product<double, ColMajor, RowMajor, ColMajor>(options, n, n, n);
    else if (options.name == "construct_col_mixed") {
        (void)coefficient_count(n, n);
        product_case<true>(
          options, Matrix<double, Dynamic, Dynamic, ColMajor>(), Matrix<double, Dynamic, Dynamic, ColMajor>(n, n),
          Matrix<double, Dynamic, Dynamic>(n, n));
    } else if (options.name == "rectangular" || options.name == "odd") {
        fdapde_strong_assert(
          n <= (std::numeric_limits<int>::max() - 3) / 2, std::invalid_argument, "rectangular size overflows");
        dynamic_product<double, RowMajor, RowMajor, RowMajor>(
          options, options.name == "odd" ? n + 1 : 2 * n + 1, n + 3, n / 2 + 1);
    } else if (options.name == "views") {
        const int count = coefficient_count(n, n);
        std::vector<double> output(std::size_t(count) + 3), lhs(std::size_t(count) + 3), rhs(std::size_t(count) + 3);
        using View = MatrixView<double, Dynamic, Dynamic>;
        product_case(
          options, View(unaligned_data(output), n, n), View(unaligned_data(lhs), n, n),
          View(unaligned_data(rhs), n, n));
    } else if (options.name == "fem3") {
        product_case(options, Matrix<double, 3, 3>(), Matrix<double, 3, 3>(), Matrix<double, 3, 3>());
    } else if (options.name == "fem4") {
        product_case(
          options, Matrix<double, 4, 4, ColMajor>(), Matrix<double, 4, 4, ColMajor>(),
          Matrix<double, 4, 4, ColMajor>());
    } else if (options.name == "fem10") {
        product_case(options, Matrix<double, 10, 10>(), Matrix<double, 10, 10>(), Matrix<double, 10, 10>());
    } else
        throw std::invalid_argument("unknown product case");
}

}   // namespace

/// @brief emits one verified measurement row or one JSON diagnostic with a failing process status
int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.suite == "assignment")
            run_assignment(options);
        else
            run_product(options);
        return 0;
    } catch (const std::exception& error) {
        std::printf("{\"verified\":false,\"warnings\":[],\"error\":");
        json_string(error.what());
        std::printf("}\n");
        return 1;
    }
}
