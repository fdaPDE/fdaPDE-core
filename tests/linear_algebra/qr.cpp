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
#include <gtest/gtest.h>

#include <limits>
#include <type_traits>
#include <utility>

namespace {

using namespace fdapde;

template <typename Decomposition>
concept permits_rvalue_q_factor = requires(Decomposition decomposition) { std::move(decomposition).Q(); };

template <typename Decomposition>
concept permits_rvalue_r_factor = requires(Decomposition decomposition) { std::move(decomposition).R(); };

using fixed_qr = HouseholderQR<double, 3, 2>;
// a QR object may exist before any matrix has been factorized
static_assert(std::is_default_constructible_v<fixed_qr>);
// the orthogonal factor cannot be borrowed from a temporary decomposition
static_assert(!permits_rvalue_q_factor<fixed_qr>);
// the triangular factor cannot be borrowed from a temporary decomposition
static_assert(!permits_rvalue_r_factor<fixed_qr>);
// a completed QR exposes an immutable owning orthogonal factor with order equal to the input row count
static_assert(std::is_same_v<decltype(std::declval<const fixed_qr&>().Q()), const OrthogonalMatrix<double, 3>&>);
// runtime row counts remain the single dynamic order of the orthogonal factor
static_assert(
  std::is_same_v<
    decltype(std::declval<const HouseholderQR<double, Dynamic, 2>&>().Q()), const OrthogonalMatrix<double, Dynamic>&>);

template <typename MatrixType, typename Decomposition>
void expect_qr_reconstruction(
  const MatrixType& source, const Decomposition& decomposition, double tolerance = 1.0e-12) {
    // successful factorization must publish completed factors before they are inspected
    ASSERT_TRUE(decomposition.computed());
    // q has one row for each input row
    ASSERT_EQ(decomposition.Q().rows(), source.rows());
    // full QR produces a square Q, including the orthogonal complement
    ASSERT_EQ(decomposition.Q().cols(), source.rows());
    // r preserves the input row count
    ASSERT_EQ(decomposition.R().rows(), source.rows());
    // r preserves the input column count
    ASSERT_EQ(decomposition.R().cols(), source.cols());

    const Matrix<double, Dynamic, Dynamic> dense_source(source);
    const Matrix<double, Dynamic, Dynamic> reconstructed(decomposition.Q() * decomposition.R());
    const double scale = dense_source.norm();
    const double denominator = scale > 0.0 ? scale : 1.0;
    // the product Q * R reconstructs the input within the scaled residual tolerance
    EXPECT_LT((reconstructed - dense_source).norm() / denominator, tolerance);

    const Matrix<double, Dynamic, Dynamic> q(decomposition.Q());
    Matrix<double, Dynamic, Dynamic> identity(q.rows(), q.rows());
    identity.set_zero();
    for (int i = 0; i < identity.rows(); ++i) identity(i, i) = 1.0;
    // the product Q.transpose() * Q matches identity within the orthogonality tolerance
    EXPECT_LT((q.transpose() * q - identity).norm(), tolerance);
    const auto& orthogonal = decomposition.Q();
    // the structured inverse composes with Q to identity for every tested shape and rank
    EXPECT_LT((orthogonal.inv() * orthogonal - identity).norm(), tolerance);
}

template <int StorageOrder> void check_qr_shapes_lifetime_and_reconstruction() {
    const Matrix<double, 4, 3, StorageOrder> tall({1.0, 2.0, -1.0, 2.0, 0.0, 3.0, -1.0, 4.0, 2.0, 3.0, -2.0, 1.0});
    const HouseholderQR tall_qr(tall);
    // deduction retains the fixed four-by-three input shape
    static_assert(decltype(tall_qr)::Rows == 4 && decltype(tall_qr)::Cols == 3);
    // the tall full-column-rank matrix has rank three
    EXPECT_EQ(tall_qr.rank(), 3);
    expect_qr_reconstruction(tall, tall_qr);

    const Matrix<double, 2, 3, StorageOrder> wide_source({1.0, -2.0, 3.0, 4.0, 1.0, -1.0});
    const Matrix<double, Dynamic, Dynamic, StorageOrder> wide(wide_source);
    const HouseholderQR wide_qr(wide);
    // a fully dynamic input produces a fully dynamic QR type
    static_assert(decltype(wide_qr)::Rows == Dynamic && decltype(wide_qr)::Cols == Dynamic);
    // the wide matrix has two independent rows
    EXPECT_EQ(wide_qr.rank(), 2);
    expect_qr_reconstruction(wide, wide_qr);

    const Matrix<double, 3, 2, StorageOrder> partial_source({1.0, 2.0, 3.0, -1.0, 2.0, 4.0});
    const Matrix<double, Dynamic, 2, StorageOrder> partial(partial_source);
    const HouseholderQR partial_qr(partial);
    // a partially dynamic input retains its fixed two columns
    static_assert(decltype(partial_qr)::Rows == Dynamic && decltype(partial_qr)::Cols == 2);
    // the partially dynamic example has two independent columns
    EXPECT_EQ(partial_qr.rank(), 2);
    expect_qr_reconstruction(partial, partial_qr);

    const MatrixView<const double, 3, 2, StorageOrder> const_view(partial_source.data());
    const HouseholderQR const_view_qr(const_view);
    // factorization of a const view owns mutable double coefficients
    static_assert(std::is_same_v<typename decltype(const_view_qr)::Scalar, double>);
    expect_qr_reconstruction(const_view, const_view_qr);

    const auto retained_qr = [] {
        const Matrix<double, 3, 2, StorageOrder> source({1.0, 2.0, 3.0, -1.0, 2.0, 4.0});
        const Matrix<double, 3, 2, StorageOrder> zero = Matrix<double, 3, 2, StorageOrder>::Zero();
        return HouseholderQR(source + zero);
    }();
    expect_qr_reconstruction(partial_source, retained_qr);
}

template <int StorageOrder> void check_qr_rank_and_scale_contracts() {
    const Matrix<double, 3, 2, StorageOrder> zero = Matrix<double, 3, 2, StorageOrder>::Zero();
    const HouseholderQR zero_qr(zero);
    // the zero matrix has rank zero
    EXPECT_EQ(zero_qr.rank(), 0);
    expect_qr_reconstruction(zero, zero_qr);

    const Matrix<double, 4, 3, StorageOrder> rank_deficient(
      {1.0, 2.0, 3.0, 2.0, 4.0, 1.0, 3.0, 6.0, -1.0, 4.0, 8.0, 2.0});
    const HouseholderQR rank_deficient_qr(rank_deficient);
    // the dependent-column example has only two independent directions
    EXPECT_EQ(rank_deficient_qr.rank(), 2);
    expect_qr_reconstruction(rank_deficient, rank_deficient_qr);

    const Matrix<double, 3, 3, StorageOrder> leading_zero_column({0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
    const HouseholderQR shifted_qr(leading_zero_column);
    // rank detection finds pivots beyond an initially zero column
    EXPECT_EQ(shifted_qr.rank(), 2);
    expect_qr_reconstruction(leading_zero_column, shifted_qr);

    for (const double scale : {1.0e-150, 1.0e150}) {
        const Matrix<double, 3, 2, StorageOrder> matrix({scale, 2.0 * scale, -3.0 * scale, scale, 2.0 * scale, -scale});
        const HouseholderQR qr(matrix);
        // uniform scaling preserves rank two over the tested magnitudes
        EXPECT_EQ(qr.rank(), 2);
        expect_qr_reconstruction(matrix, qr);
    }

    const Matrix<double, 2, 2, StorageOrder> mixed_scale({1.0, 0.0, 0.0, 1.0e-160});
    const HouseholderQR mixed_scale_qr(mixed_scale);
    // a pivot below the relative threshold yields numerical rank one
    EXPECT_EQ(mixed_scale_qr.rank(), 1);
    expect_qr_reconstruction(mixed_scale, mixed_scale_qr);
}

template <int StorageOrder> void check_qr_invalid_input_contracts() {
    HouseholderQR<double, Dynamic, Dynamic> reusable;
    // a default QR object reports no completed factorization
    EXPECT_FALSE(reusable.computed());
    // an uncomputed QR object reports rank zero
    EXPECT_EQ(reusable.rank(), 0);
    // unavailable factors are rejected before dereferencing storage for the optional orthogonal owner
    EXPECT_THROW(static_cast<void>(reusable.Q()), std::logic_error);

    const Matrix<double, 2, 2, StorageOrder> valid({1.0, 2.0, 3.0, 4.0});
    reusable.compute(valid);
    // recomputing with valid data publishes a completed factorization
    EXPECT_TRUE(reusable.computed());
    // recomputing with full-rank data updates the rank to two
    EXPECT_EQ(reusable.rank(), 2);
    const OrthogonalMatrix<double, Dynamic> previous_q(reusable.Q());
    reusable.compute(reusable.Q());
    // recomputation from the borrowed Q factor snapshots its coefficients before replacing that factor
    expect_qr_reconstruction(previous_q, reusable);

    Matrix<double, Dynamic, Dynamic, StorageOrder> nonfinite(valid);
    nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
    // a NaN coefficient is rejected as invalid QR input
    EXPECT_THROW(reusable.compute(nonfinite), std::invalid_argument);
    // rejected NaN input clears the completed state
    EXPECT_FALSE(reusable.computed());
    // rejected NaN input clears the previous rank
    EXPECT_EQ(reusable.rank(), 0);
    // failed recomputation cannot expose a stale or partially built orthogonal factor
    EXPECT_THROW(static_cast<void>(reusable.Q()), std::logic_error);

    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    // an infinite coefficient is rejected as invalid QR input
    EXPECT_THROW(reusable.compute(nonfinite), std::invalid_argument);
    // rejected infinite input leaves no completed factorization
    EXPECT_FALSE(reusable.computed());
    // rejected infinite input leaves rank zero
    EXPECT_EQ(reusable.rank(), 0);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> empty;
    // an empty matrix cannot be factorized
    EXPECT_THROW(reusable.compute(empty), std::invalid_argument);
    // rejected empty input clears the completed state
    EXPECT_FALSE(reusable.computed());
    // rejected empty input clears the previous rank
    EXPECT_EQ(reusable.rank(), 0);

    HouseholderQR<double, 3, 2> fixed_shape;
    Matrix<double, Dynamic, Dynamic, StorageOrder> wrong_shape(2, 3);
    // runtime dimensions must agree with the QR type's fixed dimensions
    EXPECT_THROW(fixed_shape.compute(wrong_shape), std::invalid_argument);
    // a fixed-shape mismatch leaves the factorization unavailable
    EXPECT_FALSE(fixed_shape.computed());
    // a fixed-shape mismatch leaves rank zero
    EXPECT_EQ(fixed_shape.rank(), 0);
}

// checks full QR reconstruction, orthogonality, rank thresholds and state reset after rejected input
TEST(linear_algebra, householder_qr) {
    check_qr_shapes_lifetime_and_reconstruction<RowMajor>();
    check_qr_shapes_lifetime_and_reconstruction<ColMajor>();
    check_qr_rank_and_scale_contracts<RowMajor>();
    check_qr_rank_and_scale_contracts<ColMajor>();
    check_qr_invalid_input_contracts<RowMajor>();
    check_qr_invalid_input_contracts<ColMajor>();
}

// a noisy line fit has an independently known minimizer and a residual orthogonal to both design columns
template <int StorageOrder> void check_qr_least_squares_solve() {
    const Matrix<double, 4, 2, StorageOrder> A({1., 0., 1., 1., 1., 2., 1., 3.});
    const MatrixView<const double, 4, 2, StorageOrder> view(A.data());
    const HouseholderQR qr(view);
    const Vector<double, 4> b({1., 2., 2., 4.});
    const auto x = qr.solve(b);
    // vector solutions have one coefficient per statically known input column
    static_assert(std::same_as<std::remove_cvref_t<decltype(x)>, Vector<double, 2>>);
    const Vector<double, 2> expected {.9, .9};
    // the least-squares intercept and slope agree with the analytic line-fit coefficients
    EXPECT_LT((x - expected).norm(), 1.e-12);
    const Vector<double, 4> residual(A * x - b);
    // the nonzero residual norm agrees with the analytic sum of squared deviations
    EXPECT_NEAR(residual.norm(), std::sqrt(.7), 1.e-12);
    // stationarity requires the residual to be orthogonal to every design column
    EXPECT_LT((A.transpose() * residual).norm(), 1.e-12);
    const Matrix<double, 4, 2> B({1., -1., 2., -.5, 2., 0., 4., .5});
    const auto X = qr.solve(B);
    // multiple right-hand sides preserve the number of output columns
    static_assert(std::same_as<std::remove_cvref_t<decltype(X)>, Matrix<double, 2, 2>>);
    const Matrix<double, 2, 2> expected_matrix({.9, -1., .9, .5});
    // both columns recover their independently known noisy and exact coefficients
    EXPECT_LT((X - expected_matrix).norm(), 1.e-12);
}

// row-major and column-major views yield the same unique least-squares minimizer for vector and matrix loads
TEST(QRSolve, LeastSquaresFromDenseViews) {
    check_qr_least_squares_solve<RowMajor>();
    check_qr_least_squares_solve<ColMajor>();
}

// square systems and dynamic shapes return independent solutions with the input column order
TEST(QRSolve, SquareDynamicAndPartialShapes) {
    const Matrix<double, 2, 2> A({2., 1., 1., 3.});
    const Vector<double, 2> expected {1., -2.};
    const Vector<double, 2> b(A * expected);
    const HouseholderQR qr(A);
    const auto x = qr.solve(b);
    // square full-rank systems recover the exact solution through the existing factors
    EXPECT_LT((x - expected).norm(), 1.e-12);
    const Vector<double, 2> zero = Vector<double, 2>::Zero();
    const auto zero_solution = qr.solve(zero);
    // a zero load returns zero coefficients without dividing by a zero right-hand-side scale
    EXPECT_EQ(zero_solution, zero);
    const Matrix<double, Dynamic, 2> partial(A);
    const HouseholderQR partial_qr(partial);
    const auto partial_x = partial_qr.solve(b);
    // fixed input columns preserve fixed solution rows even when input rows are dynamic
    static_assert(std::same_as<std::remove_cvref_t<decltype(partial_x)>, Vector<double, 2>>);
    // partially dynamic systems retain the square solve behavior
    EXPECT_LT((partial_x - expected).norm(), 1.e-12);
    const Matrix<double, Dynamic, Dynamic> dynamic(A);
    const HouseholderQR dynamic_qr(dynamic);
    const Matrix<double, Dynamic, Dynamic> dynamic_b(b);
    const auto dynamic_x = dynamic_qr.solve(dynamic_b);
    // fully dynamic matrix loads produce fully dynamic solution metadata
    static_assert(std::same_as<std::remove_cvref_t<decltype(dynamic_x)>, Matrix<double, Dynamic, Dynamic>>);
    // a runtime one-column load returns one output column
    EXPECT_EQ(dynamic_x.cols(), 1);
    // the row count follows the matrix's number of unknowns
    EXPECT_EQ(dynamic_x.rows(), 2);
    const Vector<double, 2> dynamic_vector(dynamic_x);
    // runtime one-column loads recover the same independently specified solution
    EXPECT_LT((dynamic_vector - expected).norm(), 1.e-12);
    const auto retained = [] {
        const Matrix<double, 2, 2> source({2., 1., 1., 3.});
        const Vector<double, 2> load {0., -5.};
        return HouseholderQR(source).solve(load);
    }();
    // the returned solution survives destruction of the source and temporary decomposition
    EXPECT_LT((retained - expected).norm(), 1.e-12);
}

// scalar precision and extreme scales are preserved without overflowing the projected right-hand side
TEST(QRSolve, ScalarAndScaleContracts) {
    const Matrix<float, 3, 2> A({1.f, 0.f, 0.f, 1.f, 1.f, 1.f});
    const HouseholderQR qr(A);
    const Vector<float, 3> b {1.f, 2.f, 3.f};
    const auto x = qr.solve(b);
    // float input factors return float-valued coefficients
    static_assert(std::same_as<std::remove_cvref_t<decltype(x)>, Vector<float, 2>>);
    const Vector<float, 2> expected {1.f, 2.f};
    // the consistent float system agrees with its known coefficients within float precision
    EXPECT_LT((x - expected).norm(), 1.e-5f);
    const Matrix<double, 3, 2> base({1., 2., -3., 1., 2., -1.});
    const Vector<double, 2> coefficients {.5, -1.};
    for (const double scale : {1.e-150, 1.e150}) {
        const Matrix<double, 3, 2> scaled(scale * base);
        const Vector<double, 3> load(scaled * coefficients);
        const HouseholderQR scaled_qr(scaled);
        const auto solution = scaled_qr.solve(load);
        // uniform matrix and load scaling leaves the independently specified solution unchanged
        EXPECT_LT((solution - coefficients).norm(), 1.e-12);
    }
    const Vector<double, 3> large_column {1.e308, 1.e308, 1.e308};
    const HouseholderQR large_qr(large_column);
    const Vector<double, 3> large_load {1.5e308, 1.5e308, 1.5e308};
    // normalized projection handles Q-transpose*b exceeding double range while the solution remains representable
    EXPECT_NEAR(large_qr.solve(large_load)[0], 1.5, 1.e-12);
}

// invalid shape, rank, state and coefficients fail before unavailable or unsuitable factors are consumed
TEST(QRSolve, RejectsUnsupportedSystemsAndRightHandSides) {
    HouseholderQR<double, 3, 2> empty;
    const Vector<double, 3> b {1., 2., 3.};
    // an uncomputed decomposition cannot solve a system
    EXPECT_THROW(empty.solve(b), std::logic_error);
    const Matrix<double, 3, 2> dependent({1., 2., 2., 4., 3., 6.});
    const HouseholderQR deficient(dependent);
    // a rank-deficient least-squares problem does not have the unique solution promised by this API
    EXPECT_THROW(deficient.solve(b), std::domain_error);
    const Matrix<double, 2, 3> wide({1., 0., 0., 0., 1., 0.});
    const HouseholderQR wide_qr(wide);
    const Vector<double, 2> wide_b {1., 2.};
    // underdetermined systems require a different solution contract
    EXPECT_THROW(wide_qr.solve(wide_b), std::invalid_argument);
    const Matrix<double, 3, 2> A({1., 0., 0., 1., 1., 1.});
    HouseholderQR qr(A);
    const Vector<double, Dynamic> wrong(std::vector<double> {1., 2.});
    // runtime right-hand-side rows must match the input observation count
    EXPECT_THROW(qr.solve(wrong), std::invalid_argument);
    const Matrix<double, 3, Dynamic> zero_columns(3, 0);
    // an empty set of right-hand sides is rejected at the public boundary
    EXPECT_THROW(qr.solve(zero_columns), std::invalid_argument);
    const Vector<double, 3> nonfinite {1., 2., std::numeric_limits<double>::quiet_NaN()};
    // every observation must be finite even when its direction is outside the projected basis
    EXPECT_THROW(qr.solve(nonfinite), std::invalid_argument);
    const Matrix<double, 2, 2> near_singular({1., 0., 0., 1.e-160});
    const HouseholderQR numerical_deficient(near_singular);
    // the solve uses the factorization's numerical-rank threshold rather than accepting arbitrarily small pivots
    EXPECT_THROW(numerical_deficient.solve(wide_b), std::domain_error);
    Matrix<double, 3, 2> invalid(A);
    invalid(0, 0) = std::numeric_limits<double>::infinity();
    // failed recomputation clears the state used by solve
    EXPECT_THROW(qr.compute(invalid), std::invalid_argument);
    // solve cannot reuse stale factors after the replacement input is rejected
    EXPECT_THROW(qr.solve(b), std::logic_error);
    const double tiny_data[] = {1.e-308};
    const Matrix<double, 1, 1> tiny(tiny_data);
    const HouseholderQR tiny_qr(tiny);
    const Vector<double, 1> huge_load {1.e308};
    // a mathematically finite quotient outside the output scalar range raises a numerical error
    EXPECT_THROW(tiny_qr.solve(huge_load), std::domain_error);
    const double largest = std::numeric_limits<double>::max();
    const Vector<double, 2> large_column {largest, largest};
    const HouseholderQR unrepresentable(large_column);
    // overflowed public triangular factors cannot silently yield a spurious finite solution
    EXPECT_THROW(unrepresentable.solve(wide_b), std::domain_error);
}

}   // namespace
