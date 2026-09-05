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
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef FDAPDE_TEST_DATA_DIR
#    error "FDAPDE_TEST_DATA_DIR must identify the test fixture directory"
#endif

namespace {

using sparse_matrix = fdapde::SparseMatrix<double>;
using float_fspai = fdapde::FSPAI<float>;

template <typename T>
concept has_rvalue_lower_factor = requires(T&& value) { std::move(value).getL(); };

static_assert(std::is_same_v<typename fdapde::FSPAI<double>::Scalar, double>);
static_assert(std::is_same_v<typename float_fspai::MatrixType, fdapde::SparseMatrix<float>>);
static_assert(std::is_same_v<typename float_fspai::StorageIndex, int>);
static_assert(
  std::is_same_v<typename float_fspai::DenseMatrixType, fdapde::Matrix<float, fdapde::Dynamic, fdapde::Dynamic>>);
static_assert(std::is_same_v<typename float_fspai::DenseVectorType, fdapde::Vector<float, fdapde::Dynamic>>);
static_assert(std::is_same_v<typename float_fspai::MatrixL, fdapde::SparseMatrix<float>>);
static_assert(std::is_same_v<typename float_fspai::MatrixU, fdapde::SparseMatrix<float>>);
static_assert(
  std::is_same_v<decltype(std::declval<const float_fspai&>().getL()), const float_fspai::MatrixL&>);
static_assert(std::is_same_v<decltype(std::declval<const float_fspai&>().getU()), float_fspai::MatrixU>);
static_assert(!has_rvalue_lower_factor<fdapde::FSPAI<double>>);

std::string next_market_line(std::ifstream& input) {
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.front() != '%') return line;
    }
    throw std::runtime_error("MatrixMarket input ended unexpectedly");
}

sparse_matrix read_matrix_market(const std::string& filename, bool expand_symmetric = true) {
    const std::string path = std::string(FDAPDE_TEST_DATA_DIR) + "/mtx/" + filename;
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open MatrixMarket fixture: " + path);

    std::string banner;
    if (!std::getline(input, banner)) throw std::runtime_error("MatrixMarket fixture has no banner");
    std::istringstream banner_stream(banner);
    std::string magic, object, format, field, symmetry;
    banner_stream >> magic >> object >> format >> field >> symmetry;
    if (
      magic != "%%MatrixMarket" || object != "matrix" || format != "coordinate" || field != "real" ||
      (symmetry != "general" && symmetry != "symmetric")) {
        throw std::runtime_error("unsupported MatrixMarket fixture format");
    }

    std::istringstream shape_stream(next_market_line(input));
    int rows = 0, cols = 0, stored = 0;
    if (!(shape_stream >> rows >> cols >> stored) || rows < 0 || cols < 0 || stored < 0) {
        throw std::runtime_error("invalid MatrixMarket fixture shape");
    }
    std::vector<fdapde::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(stored) * (symmetry == "symmetric" ? 2 : 1));
    for (int current = 0; current < stored; ++current) {
        std::istringstream entry_stream(next_market_line(input));
        int row = 0, col = 0;
        double value = 0.0;
        if (
          !(entry_stream >> row >> col >> value) || row < 1 || row > rows || col < 1 || col > cols ||
          !std::isfinite(value)) {
            throw std::runtime_error("invalid MatrixMarket fixture entry");
        }
        triplets.emplace_back(row - 1, col - 1, value);
        if (expand_symmetric && symmetry == "symmetric" && row != col) {
            triplets.emplace_back(col - 1, row - 1, value);
        }
    }
    return sparse_matrix(rows, cols, triplets);
}

double max_abs_difference(const sparse_matrix& lhs, const sparse_matrix& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("sparse comparison requires equal shapes");
    }
    double result = 0.0;
    for (int row = 0; row < lhs.rows(); ++row) {
        for (const auto entry : lhs.row(row)) {
            result = std::max(result, std::abs(entry.value() - rhs.coeff(row, entry.column())));
        }
        for (const auto entry : rhs.row(row)) {
            result = std::max(result, std::abs(entry.value() - lhs.coeff(row, entry.column())));
        }
    }
    return result;
}

double factor_product_max_abs_difference(const sparse_matrix& actual, const sparse_matrix& factor) {
    if (
      actual.rows() != actual.cols() || factor.rows() != factor.cols() || actual.rows() != factor.rows() ||
      actual.cols() != factor.cols()) {
        throw std::invalid_argument("factor product requires equal square shapes");
    }
    const int n = factor.rows();
    std::vector<double> dense(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int row = 0; row < n; ++row) {
        for (const auto entry : factor.row(row)) {
            dense
              [static_cast<std::size_t>(row) * static_cast<std::size_t>(n) + static_cast<std::size_t>(entry.column())] =
                entry.value();
        }
    }

    double result = 0.0;
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            double expected = 0.0;
            for (int k = 0; k < n; ++k) {
                expected +=
                  dense[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) + static_cast<std::size_t>(k)] *
                  dense[static_cast<std::size_t>(col) * static_cast<std::size_t>(n) + static_cast<std::size_t>(k)];
            }
            result = std::max(result, std::abs(actual.coeff(row, col) - expected));
        }
    }
    return result;
}

void expect_same_sparse(const sparse_matrix& lhs, const sparse_matrix& rhs) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    ASSERT_EQ(lhs.non_zeros(), rhs.non_zeros());
    for (int row = 0; row < lhs.rows(); ++row) {
        auto lhs_entry = lhs.row(row).begin();
        auto rhs_entry = rhs.row(row).begin();
        const auto lhs_end = lhs.row(row).end();
        const auto rhs_end = rhs.row(row).end();
        while (lhs_entry != lhs_end && rhs_entry != rhs_end) {
            EXPECT_EQ((*lhs_entry).column(), (*rhs_entry).column());
            EXPECT_DOUBLE_EQ((*lhs_entry).value(), (*rhs_entry).value());
            ++lhs_entry;
            ++rhs_entry;
        }
        EXPECT_EQ(lhs_entry, lhs_end);
        EXPECT_EQ(rhs_entry, rhs_end);
    }
}

template <typename Lhs, typename Rhs> void expect_same_dense(const Lhs& lhs, const Rhs& rhs) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (int row = 0; row < lhs.rows(); ++row) {
        for (int col = 0; col < lhs.cols(); ++col) { EXPECT_DOUBLE_EQ(lhs(row, col), rhs(row, col)); }
    }
}

TEST(FspaiTestSuite, NativeAliasesSupportFloatScalars) {
    const float_fspai::MatrixType source(
      2, 2,
      {
        {0, 0, 4.0f},
        {1, 1, 9.0f}
    });
    const fdapde::FSPAI default_parameters(source);
    const fdapde::FSPAI fspai(source, 0, 0, 0.0);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(default_parameters)>, float_fspai>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(fspai)>, float_fspai>);
    EXPECT_EQ(default_parameters.rows(), 2);

    const float_fspai::MatrixL lower = fspai.getL();
    const float_fspai::MatrixU upper = fspai.getU();
    EXPECT_FLOAT_EQ(lower.coeff(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(lower.coeff(1, 1), 1.0f / 3.0f);
    EXPECT_FLOAT_EQ(upper.coeff(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(upper.coeff(1, 1), 1.0f / 3.0f);

    float_fspai::DenseVectorType rhs(2);
    rhs[0] = 4.0f;
    rhs[1] = 9.0f;
    const float_fspai::DenseMatrixType solved = fspai.solve(rhs);
    EXPECT_FLOAT_EQ(solved(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(solved(1, 0), 1.0f);
}

// Adapted from a2a9c88:test/src/fspai_test.cpp. The archived loader ignored
// the symmetric banner, so this fixture is the stored lower factor.
// The approved inverse oracle compares the native result with an independent
// test-local dense factor product.
TEST(FspaiTestSuite, FspaiTest) {
    const sparse_matrix source = read_matrix_market("matrix_to_be_inverted.mtx");
    const sparse_matrix expected_factor = read_matrix_market("expected_inverted_matrix.mtx", false);

    fdapde::FSPAI<double> fspai;
    fspai.compute(source, 10, 10, 0.005);
    const sparse_matrix actual = fspai.inverse();

    EXPECT_EQ(fspai.rows(), 264);
    EXPECT_EQ(fspai.cols(), 264);
    EXPECT_LT(max_abs_difference(fspai.getL(), expected_factor), 1.0e-7);
    EXPECT_LT(factor_product_max_abs_difference(actual, expected_factor), 1.0e-7);
}

TEST(FspaiTestSuite, FactorsAreOwningOrientedAndCopySafe) {
    const sparse_matrix source(
      3, 3,
      {
        {0, 0, 4.0 },
        {0, 1, 1.0 },
        {0, 2, 0.5 },
        {1, 0, 1.0 },
        {1, 1, 3.0 },
        {1, 2, 0.25},
        {2, 0, 0.5 },
        {2, 1, 0.25},
        {2, 2, 2.0 }
    });
    fdapde::FSPAI<double> fspai(source, 2, 2, 0.0);
    const sparse_matrix lower = fspai.getL();
    const sparse_matrix upper = fspai.getU();
    const sparse_matrix inverse = fspai.inverse();

    for (int row = 0; row < lower.rows(); ++row) {
        for (const auto entry : lower.row(row)) EXPECT_GE(row, entry.column());
        for (const auto entry : upper.row(row)) EXPECT_LE(row, entry.column());
        for (int col = 0; col < lower.cols(); ++col) { EXPECT_DOUBLE_EQ(upper.coeff(row, col), lower.coeff(col, row)); }
    }
    expect_same_sparse(inverse, lower * upper);

    fdapde::FSPAI<double> copy(fspai);
    copy.compute(
      sparse_matrix(
        2, 2,
        {
          {0, 0, 9.0 },
          {1, 1, 16.0}
    }),
      0, 0, 0.0);
    EXPECT_EQ(copy.rows(), 2);
    EXPECT_EQ(fspai.rows(), 3);
    expect_same_sparse(fspai.getL(), lower);

    fdapde::FSPAI<double> moved(std::move(copy));
    EXPECT_EQ(moved.rows(), 2);
    EXPECT_EQ(copy.rows(), 0);
    EXPECT_EQ(copy.cols(), 0);
    EXPECT_THROW(static_cast<void>(copy.getL()), std::domain_error);
    fdapde::FSPAI<double> move_assigned;
    move_assigned = std::move(moved);
    EXPECT_EQ(move_assigned.rows(), 2);
    EXPECT_EQ(moved.rows(), 0);
    EXPECT_EQ(moved.cols(), 0);
    EXPECT_THROW(static_cast<void>(moved.inverse()), std::domain_error);

    fdapde::FSPAI<double> diagonal_only(source, 0, 0, 0.0);
    EXPECT_EQ(diagonal_only.getL().non_zeros(), 3);
    EXPECT_DOUBLE_EQ(diagonal_only.getL().coeff(0, 0), 0.5);
    EXPECT_DOUBLE_EQ(diagonal_only.getL().coeff(1, 1), 1.0 / std::sqrt(3.0));
    EXPECT_DOUBLE_EQ(diagonal_only.getL().coeff(2, 2), 1.0 / std::sqrt(2.0));
}

// Native adaptation of the archived FSPAI implementation at 86ff6d12; final behavior introduced by da0b274:
// all solve overloads apply L * L.transpose().
TEST(FspaiTestSuite, SolvesDenseAndSparseRightHandSidesByTheCompleteApproximateInverse) {
    const sparse_matrix system(
      2, 2,
      {
        {0, 0, 4.0},
        {1, 1, 9.0}
    });
    const fdapde::FSPAI<double> fspai(system, 0, 0, 0.0);

    fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> dense_rhs(2, 3);
    dense_rhs(0, 0) = 4.0;
    dense_rhs(0, 1) = 8.0;
    dense_rhs(0, 2) = 12.0;
    dense_rhs(1, 0) = 9.0;
    dense_rhs(1, 1) = 18.0;
    dense_rhs(1, 2) = 27.0;
    const auto retained_dense_rhs = dense_rhs;
    const auto dense_expected = fspai.inverse() * dense_rhs;
    const auto dense_result = fspai.solve(dense_rhs);
    expect_same_dense(dense_result, dense_expected);
    expect_same_dense(dense_rhs, retained_dense_rhs);

    fdapde::Vector<double, fdapde::Dynamic> vector_rhs(2);
    vector_rhs[0] = 8.0;
    vector_rhs[1] = 27.0;
    const auto vector_expected = fspai.inverse() * vector_rhs;
    const auto vector_result = fspai.solve(vector_rhs);
    expect_same_dense(vector_result, vector_expected);

    const sparse_matrix sparse_rhs(
      2, 3,
      {
        {0, 0, 4.0 },
        {0, 2, 12.0},
        {1, 1, 18.0}
    });
    const sparse_matrix sparse_expected = fspai.inverse() * sparse_rhs;
    const sparse_matrix sparse_result = fspai.solve(sparse_rhs);
    expect_same_sparse(sparse_result, sparse_expected);

    auto dense_in_place = dense_rhs;
    fspai.solveInPlace(dense_in_place);
    expect_same_dense(dense_in_place, dense_expected);
    auto sparse_in_place = sparse_rhs;
    fspai.solveInPlace(sparse_in_place);
    expect_same_sparse(sparse_in_place, sparse_expected);
}

TEST(FspaiTestSuite, SolveFailuresRemainCheckedAndInPlaceUpdatesAreAtomic) {
    fdapde::FSPAI<double> unready;
    fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> dense_rhs(2, 1);
    dense_rhs(0, 0) = 4.0;
    dense_rhs(1, 0) = 9.0;
    const auto retained_dense = dense_rhs;
    EXPECT_THROW(static_cast<void>(unready.solve(dense_rhs)), std::domain_error);
    EXPECT_THROW(unready.solveInPlace(dense_rhs), std::domain_error);
    expect_same_dense(dense_rhs, retained_dense);

    const fdapde::FSPAI<double> fspai(
      sparse_matrix(
        2, 2,
        {
          {0, 0, 4.0},
          {1, 1, 9.0}
    }),
      0, 0, 0.0);
    fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> wrong_dense(3, 1);
    const auto retained_wrong_dense = wrong_dense;
    EXPECT_THROW(static_cast<void>(fspai.solve(wrong_dense)), std::invalid_argument);
    EXPECT_THROW(fspai.solveInPlace(wrong_dense), std::invalid_argument);
    expect_same_dense(wrong_dense, retained_wrong_dense);

    dense_rhs(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(static_cast<void>(fspai.solve(dense_rhs)), std::invalid_argument);
    EXPECT_THROW(fspai.solveInPlace(dense_rhs), std::invalid_argument);
    EXPECT_TRUE(std::isinf(dense_rhs(0, 0)));

    sparse_matrix sparse_rhs(
      2, 1,
      {
        {0, 0, 4.0},
        {1, 0, 9.0}
    });
    const sparse_matrix retained_sparse = sparse_rhs;
    sparse_matrix wrong_sparse(
      3, 1,
      {
        {0, 0, 4.0}
    });
    const sparse_matrix retained_wrong_sparse = wrong_sparse;
    EXPECT_THROW(static_cast<void>(fspai.solve(wrong_sparse)), std::invalid_argument);
    EXPECT_THROW(fspai.solveInPlace(wrong_sparse), std::invalid_argument);
    expect_same_sparse(wrong_sparse, retained_wrong_sparse);
    sparse_rhs.value_ref(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(fspai.solve(sparse_rhs)), std::invalid_argument);
    EXPECT_THROW(fspai.solveInPlace(sparse_rhs), std::invalid_argument);
    EXPECT_TRUE(std::isnan(sparse_rhs.coeff(0, 0)));
    expect_same_sparse(
      retained_sparse, sparse_matrix(
                         2, 1,
                         {
                           {0, 0, 4.0},
                           {1, 0, 9.0}
    }));
}

TEST(FspaiTestSuite, ContractsRemainActiveWithoutDebugAssertions) {
    fdapde::FSPAI<double> fspai;
    EXPECT_EQ(fspai.rows(), 0);
    EXPECT_EQ(fspai.cols(), 0);
    EXPECT_THROW(static_cast<void>(fspai.getL()), std::domain_error);
    EXPECT_THROW(static_cast<void>(fspai.getU()), std::domain_error);
    EXPECT_THROW(static_cast<void>(fspai.inverse()), std::domain_error);

    const sparse_matrix valid(
      2, 2,
      {
        {0, 0, 2.0 },
        {0, 1, 0.25},
        {1, 0, 0.25},
        {1, 1, 3.0 }
    });
    fspai.compute(valid, 2, 2, 0.0);
    const sparse_matrix retained = fspai.getL();

    const sparse_matrix scale_separated_spd(
      3, 3,
      {
        {0, 0, 1.0e0  },
        {1, 1, 1.0e-20},
        {1, 2, 1.0e-25},
        {2, 1, 1.0e-25},
        {2, 2, 1.0e-20}
    });
    EXPECT_NO_THROW((fdapde::FSPAI<double>(scale_separated_spd, 2, 2, 0.0)));

    EXPECT_THROW(fspai.compute(sparse_matrix()), std::invalid_argument);
    EXPECT_THROW(fspai.compute(sparse_matrix(2, 3)), std::invalid_argument);
    EXPECT_THROW(fspai.compute(valid, -1, 1, 0.0), std::invalid_argument);
    EXPECT_THROW(fspai.compute(valid, 1, -1, 0.0), std::invalid_argument);
    EXPECT_THROW(fspai.compute(valid, 1, 1, -0.1), std::invalid_argument);
    EXPECT_THROW(fspai.compute(valid, 1, 1, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(
      fspai.compute(sparse_matrix(
        2, 2,
        {
          {0, 0, 1.0},
          {0, 1, 0.5},
          {1, 1, 1.0}
    })),
      std::invalid_argument);
    EXPECT_THROW(
      fspai.compute(sparse_matrix(
        2, 2,
        {
          {0, 0, 1.0},
          {0, 1, 2.0},
          {1, 0, 2.0},
          {1, 1, 1.0}
    })),
      std::domain_error);
    EXPECT_THROW(
      fspai.compute(
        sparse_matrix(
          2, 2,
          {
            {0, 0, 1.0},
            {0, 1, 1.0},
            {1, 0, 1.0},
            {1, 1, 1.0}
    }),
        0, 0, 0.0),
      std::domain_error);
    EXPECT_THROW(
      fspai.compute(sparse_matrix(
        2, 2,
        {
          {0, 0, 1.0                                    },
          {1, 1, std::numeric_limits<double>::infinity()}
    })),
      std::invalid_argument);
    expect_same_sparse(fspai.getL(), retained);
}

}   // namespace
