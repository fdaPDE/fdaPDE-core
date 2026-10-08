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

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace {

using namespace fdapde;

template <typename Decomposition>
concept permits_rvalue_eigenvalues = requires(Decomposition decomposition) { std::move(decomposition).eigenvalues(); };

template <typename Decomposition>
concept permits_rvalue_eigenvectors =
  requires(Decomposition decomposition) { std::move(decomposition).eigenvectors(); };

using fixed_symmetric = SymmetricMatrix<double, 3>;
using fixed_evd = EVD<fixed_symmetric>;
using const_view_evd = EVD<SymmetricMatrixView<const double, 3>>;
// an eigendecomposition may be constructed before receiving a matrix
static_assert(std::is_default_constructible_v<fixed_evd>);
// a const input view still produces owned double-valued factors
static_assert(std::is_same_v<typename const_view_evd::Scalar, double>);
// eigenvalue references cannot escape a temporary decomposition
static_assert(!permits_rvalue_eigenvalues<fixed_evd>);
// an eigenvector factor cannot borrow a temporary decomposition
static_assert(!permits_rvalue_eigenvectors<fixed_evd>);
// fixed eigenvector factors expose the native orthogonal owner through an immutable reference
static_assert(
  std::is_same_v<decltype(std::declval<const fixed_evd&>().eigenvectors()), const OrthogonalMatrix<double, 3>&>);

/// @brief advertises an overflowing dense workspace size and records any coefficient access
struct oversized_symmetric_expression : SymmetricMatrixExpr<oversized_symmetric_expression> {
    using Scalar = double;
    static constexpr int Rows = Dynamic;
    static constexpr int Cols = Dynamic;

    /// @brief binds the flag used to detect premature coefficient access
    explicit oversized_symmetric_expression(bool& coefficient_accessed) :
        coefficient_accessed_(&coefficient_accessed) { }

    /// @brief records coefficient evaluation before returning zero
    double operator()(int, int) const {
        *coefficient_accessed_ = true;
        return 0.0;
    }
    /// @brief returns a row extent whose square exceeds the supported workspace size
    constexpr int rows() const { return 46341; }
    /// @brief returns the matching column extent of the oversized square expression
    constexpr int cols() const { return 46341; }
   private:
    bool* coefficient_accessed_;
};

template <typename MatrixType, typename Decomposition>
void expect_valid_evd(const MatrixType& source, const Decomposition& decomposition, double tolerance = 1.0e-10) {
    // successful eigendecomposition must publish completed factors before inspection
    ASSERT_TRUE(decomposition.computed());
    // there is one eigenvalue for each matrix row
    ASSERT_EQ(decomposition.eigenvalues().size(), source.rows());

    const Matrix<double, Dynamic, Dynamic> dense_source(source);
    const Matrix<double, Dynamic, Dynamic> eigenvectors(decomposition.eigenvectors());
    Matrix<double, Dynamic, Dynamic> diagonal(source.rows(), source.cols());
    Matrix<double, Dynamic, Dynamic> identity(source.rows(), source.cols());
    diagonal.set_zero();
    identity.set_zero();
    for (int i = 0; i < source.rows(); ++i) {
        diagonal(i, i) = decomposition.eigenvalues()[i];
        identity(i, i) = 1.0;
        // computed eigenvalues remain finite for finite test inputs
        EXPECT_TRUE(std::isfinite(decomposition.eigenvalues()[i]));
    }

    const double source_norm = dense_source.norm();
    const double denominator = source_norm > 0 ? source_norm : 1.0;
    // each eigenvector satisfies A times v equals lambda times v within scaled tolerance
    EXPECT_LT((dense_source * eigenvectors - eigenvectors * diagonal).norm() / denominator, tolerance);
    // the eigenvector columns form an orthonormal basis
    EXPECT_LT((eigenvectors.transpose() * eigenvectors - identity).norm(), tolerance);
    const auto& basis = decomposition.eigenvectors();
    // the structured inverse composes with the eigenvector basis to identity across the tested eigenspaces
    EXPECT_LT((basis.inv() * basis - identity).norm(), tolerance);
    // v times D times V transpose reconstructs the input within scaled tolerance
    EXPECT_LT((eigenvectors * diagonal * eigenvectors.transpose() - dense_source).norm() / denominator, tolerance);
}

template <int StorageOrder> void check_evd_shapes_lifetime_and_storage_order() {
    const Matrix<double, 3, 3, StorageOrder> dense({4.0, 1.0, -2.0, 1.0, 3.0, 0.5, -2.0, 0.5, 2.0});
    const auto symmetric = dense.template as_symmetric<Lower>();
    const EVD decomposition(symmetric);
    // deduction retains the fixed three-by-three input shape
    static_assert(decltype(decomposition)::Rows == 3 && decltype(decomposition)::Cols == 3);
    expect_valid_evd(symmetric, decomposition);

    const Matrix<double, 2, 2, StorageOrder> two_by_two({2.0, 1.0, 1.0, 3.0});
    const auto two_by_two_symmetric = two_by_two.template as_symmetric<Lower>();
    const EVD two_by_two_evd(two_by_two_symmetric);
    const double minimum = std::min(two_by_two_evd.eigenvalues()[0], two_by_two_evd.eigenvalues()[1]);
    const double maximum = std::max(two_by_two_evd.eigenvalues()[0], two_by_two_evd.eigenvalues()[1]);
    // the smaller eigenvalue matches the closed-form two-by-two solution
    EXPECT_NEAR(minimum, (5.0 - std::sqrt(5.0)) / 2.0, 1.0e-12);
    // the larger eigenvalue matches the closed-form two-by-two solution
    EXPECT_NEAR(maximum, (5.0 + std::sqrt(5.0)) / 2.0, 1.0e-12);
    expect_valid_evd(two_by_two_symmetric, two_by_two_evd);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> dynamic(dense);
    const auto dynamic_symmetric = dynamic.template as_symmetric<Lower>();
    const EVD dynamic_evd(dynamic_symmetric);
    // a fully dynamic symmetric input produces dynamic factor dimensions
    static_assert(decltype(dynamic_evd)::Rows == Dynamic && decltype(dynamic_evd)::Cols == Dynamic);
    expect_valid_evd(dynamic_symmetric, dynamic_evd);

    const Matrix<double, Dynamic, 3, StorageOrder> partial(dense);
    const auto partial_symmetric = partial.template as_symmetric<Lower>();
    const EVD partial_evd(partial_symmetric);
    // a partially dynamic input retains its fixed dimension
    static_assert(decltype(partial_evd)::Rows == Dynamic && decltype(partial_evd)::Cols == 3);
    // the fixed column count supplies the single order of this partially dynamic input's orthogonal factor
    static_assert(std::is_same_v<decltype(partial_evd.eigenvectors()), const OrthogonalMatrix<double, 3>&>);
    expect_valid_evd(partial_symmetric, partial_evd);

    const MatrixView<const double, 3, 3, StorageOrder> const_view(dense.data());
    const auto const_symmetric = const_view.template as_symmetric<Lower>();
    const EVD const_view_evd(const_symmetric);
    // factorization removes input-view constness from the owned scalar type
    static_assert(std::is_same_v<typename decltype(const_view_evd)::Scalar, double>);
    expect_valid_evd(const_symmetric, const_view_evd);

    const auto retained_evd = [] {
        const Matrix<double, 3, 3, StorageOrder> source({4.0, 1.0, -2.0, 1.0, 3.0, 0.5, -2.0, 0.5, 2.0});
        const Matrix<double, 3, 3, StorageOrder> zero = Matrix<double, 3, 3, StorageOrder>::Zero();
        return (source + zero).template as_symmetric<Lower>().evd();
    }();
    expect_valid_evd(symmetric, retained_evd);
}

void check_evd_repeated_scale_and_zero_contracts() {
    SymmetricMatrix<double, Dynamic> repeated(4, 4);
    for (int row = 0; row < 4; ++row) {
        for (int col = row; col < 4; ++col) repeated(row, col) = row == col ? 2.75 : 0.75;
    }
    const auto repeated_evd = repeated.evd();
    int repeated_twos = 0;
    int repeated_fives = 0;
    for (const double eigenvalue : repeated_evd.eigenvalues()) {
        repeated_twos += std::abs(eigenvalue - 2.0) < 1.0e-10;
        repeated_fives += std::abs(eigenvalue - 5.0) < 1.0e-10;
    }
    // the repeated eigenvalue two retains multiplicity three
    EXPECT_EQ(repeated_twos, 3);
    // the eigenvalue five occurs once
    EXPECT_EQ(repeated_fives, 1);
    expect_valid_evd(repeated, repeated_evd);

    for (const double scale : {1.0e-20, 1.0e150}) {
        const SymmetricMatrix<double, 2> matrix({2.0 * scale, scale, 2.0 * scale});
        const auto decomposition = matrix.evd();
        const double minimum = std::min(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]);
        const double maximum = std::max(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]);
        // uniform scaling preserves the smaller normalized eigenvalue one
        EXPECT_NEAR(minimum / scale, 1.0, 1.0e-12);
        // uniform scaling preserves the larger normalized eigenvalue three
        EXPECT_NEAR(maximum / scale, 3.0, 1.0e-12);
        expect_valid_evd(matrix, decomposition);
    }

    const SymmetricMatrix<double, 3> zero({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    const auto zero_evd = zero.evd();
    // the zero matrix produces only zero eigenvalues
    EXPECT_EQ(zero_evd.eigenvalues(), (Vector<double, 3>({0.0, 0.0, 0.0})));
    const Matrix<double, 3, 3> identity = IdentityMatrix<double, 3, 3>();
    const Matrix<double, 3, 3> zero_eigenvectors(zero_evd.eigenvectors());
    for (int i = 0; i < 3; ++i) {
        // the zero-matrix eigenvectors retain the initialized identity basis
        for (int j = 0; j < 3; ++j) EXPECT_DOUBLE_EQ(zero_eigenvectors(i, j), identity(i, j));
    }
    expect_valid_evd(zero, zero_evd);
}

void check_evd_invalid_input_contracts() {
    EVD<SymmetricMatrix<double, Dynamic>> reusable;
    // a default object has no completed eigendecomposition
    EXPECT_FALSE(reusable.computed());
    // unavailable eigenvectors cannot dereference an orthogonal owner that has not been built
    EXPECT_THROW(static_cast<void>(reusable.eigenvectors()), std::logic_error);
    // unavailable eigenvalues cannot expose a default vector as a completed spectrum
    EXPECT_THROW(static_cast<void>(reusable.eigenvalues()), std::logic_error);

    SymmetricMatrix<double, Dynamic> valid(2, 2);
    valid(0, 0) = 2.0;
    valid(1, 0) = 1.0;
    valid(1, 1) = 3.0;
    reusable.compute(valid);
    // valid recomputation publishes completed eigenpairs
    EXPECT_TRUE(reusable.computed());

    bool coefficient_accessed = false;
    const oversized_symmetric_expression oversized(coefficient_accessed);
    try {
        reusable.compute(oversized);
        // reaching this point means workspace validation failed to reject the oversized expression
        FAIL() << "oversized EVD workspace was accepted";
    } catch (const std::length_error& error) {
        // oversized workspace rejection reports the specific capacity error
        EXPECT_STREQ(error.what(), "EVD: dense workspace size exceeds supported range");
    }
    // workspace overflow is rejected before any input coefficient is read
    EXPECT_FALSE(coefficient_accessed);
    // workspace rejection clears the previously completed state
    EXPECT_FALSE(reusable.computed());
    // failed recomputation cannot expose the previous orthogonal factor
    EXPECT_THROW(static_cast<void>(reusable.eigenvectors()), std::logic_error);

    valid(0, 0) = std::numeric_limits<double>::quiet_NaN();
    // a NaN coefficient is rejected before eigenpairs are published
    EXPECT_THROW(reusable.compute(valid), std::invalid_argument);
    // a NaN input leaves no completed eigendecomposition
    EXPECT_FALSE(reusable.computed());

    valid(0, 0) = std::numeric_limits<double>::infinity();
    // an infinite coefficient is rejected before eigenpairs are published
    EXPECT_THROW(reusable.compute(valid), std::invalid_argument);
    // an infinite input leaves no completed eigendecomposition
    EXPECT_FALSE(reusable.computed());

    const SymmetricMatrix<double, Dynamic> empty;
    // an empty symmetric matrix is rejected
    EXPECT_THROW(reusable.compute(empty), std::invalid_argument);
    // empty input leaves no completed eigendecomposition
    EXPECT_FALSE(reusable.computed());

    EVD<SymmetricMatrix<double, 3>> fixed_shape;
    SymmetricMatrix<double, Dynamic> wrong_shape(2, 2);
    // runtime dimensions must match the decomposition's fixed shape
    EXPECT_THROW(fixed_shape.compute(wrong_shape), std::invalid_argument);
    // a fixed-shape mismatch leaves no completed eigendecomposition
    EXPECT_FALSE(fixed_shape.computed());
}

// checks symmetric eigenpairs, orthogonality, repeated values, scaling and invalid-input state reset
TEST(linear_algebra, symmetric_evd) {
    check_evd_shapes_lifetime_and_storage_order<RowMajor>();
    check_evd_shapes_lifetime_and_storage_order<ColMajor>();
    check_evd_repeated_scale_and_zero_contracts();
    check_evd_invalid_input_contracts();
}

}   // namespace

// generic dense inputs and views validate symmetry before using the symmetric eigensolver
TEST(EVD, GenericDenseInputsAndViews) {
    const fdapde::Matrix<double, 2, 2> A({2., 1., 1., 2.});
    fdapde::EVD decomposition(A);
    // generic matrix input reconstructs through the eigenpairs without a symmetric adaptor
    expect_valid_evd(A, decomposition);
    const fdapde::MatrixView<const double, 2, 2> view(A.data());
    const fdapde::EVD viewed(view);
    // a const generic view supplies the same mathematical eigenproblem
    expect_valid_evd(A, viewed);
    // nonsymmetric coefficients are rejected rather than mirrored from a selected triangle
    EXPECT_THROW(decomposition.compute(fdapde::Matrix<double, 2, 2>({2., 3., 1., 2.})), std::invalid_argument);
    // failed validation prevents stale eigenpairs from being consumed
    EXPECT_THROW(decomposition.eigenvectors(), std::logic_error);
    // nonfinite generic input is rejected before entering the small-order kernel
    EXPECT_THROW(
      decomposition.compute(fdapde::Matrix<double, 2, 2>({2., 1., 1., std::numeric_limits<double>::quiet_NaN()})),
      std::invalid_argument);
}

namespace {

// generic dense views preserve indefinite spectra and recover independently specified solutions
template <int StorageOrder> void check_evd_indefinite_solve() {
    const Matrix<double, 3, 3, StorageOrder> A({0., 2., 0., 2., 0., 0., 0., 0., -3.});
    const MatrixView<const double, 3, 3, StorageOrder> view(A.data());
    const EVD evd(view);
    const Matrix<double, 3, 2> expected({1., -2., 2., 0., -1., 3.});
    const Matrix<double, 3, 2> B(A * expected);
    const auto X = evd.solve(B);
    // matrix right-hand sides preserve the known order and number of output columns
    static_assert(std::same_as<std::remove_cvref_t<decltype(X)>, Matrix<double, 3, 2>>);
    // signed reciprocal eigenvalues recover the independently specified coefficients for an indefinite matrix
    EXPECT_LT((X - expected).norm(), 1.e-12);
    // residuals against the original operator validate the reconstructed solution for both loads
    EXPECT_LT((A * X - B).norm(), 1.e-12);
    const auto x = evd.solve(B.col(0));
    // a vector load produces an owning vector with the fixed input order
    static_assert(std::same_as<std::remove_cvref_t<decltype(x)>, Vector<double, 3>>);
    // a borrowed right-hand-side column yields the same first solution column
    EXPECT_LT((x - expected.col(0)).norm(), 1.e-12);
}

// both dense storage orders support indefinite systems with vector and matrix right-hand sides
TEST(EVDSolve, DenseViewsAndIndefiniteSystems) {
    check_evd_indefinite_solve<RowMajor>();
    check_evd_indefinite_solve<ColMajor>();
}

// cached and dynamic decompositions return independent solutions with the most precise available static order
TEST(EVDSolve, DynamicShapesCacheAndLifetime) {
    const Matrix<double, 2, 2> A({2., 1., 1., 2.});
    const Vector<double, 2> B {1., 0.};
    const Vector<double, 2> expected {2. / 3., -1. / 3.};
    const Matrix<double, Dynamic, 2> partial(A);
    const EVD partial_evd(partial);
    const auto partial_x = partial_evd.solve(B);
    // fixed columns preserve the factor order for partially dynamic square input
    static_assert(std::same_as<std::remove_cvref_t<decltype(partial_x)>, Vector<double, 2>>);
    // partial dynamic metadata does not change the analytic solution
    EXPECT_LT((partial_x - expected).norm(), 1.e-12);
    const Matrix<double, Dynamic, Dynamic> dynamic(A);
    const EVD dynamic_evd(dynamic);
    const Matrix<double, Dynamic, Dynamic> dynamic_B(B);
    const auto dynamic_X = dynamic_evd.solve(dynamic_B);
    // fully dynamic matrix loads retain fully dynamic solution metadata
    static_assert(std::same_as<std::remove_cvref_t<decltype(dynamic_X)>, Matrix<double, Dynamic, Dynamic>>);
    // runtime solution rows follow the operator order
    EXPECT_EQ(dynamic_X.rows(), 2);
    // a runtime one-column load retains one output column
    EXPECT_EQ(dynamic_X.cols(), 1);
    const Vector<double, 2> dynamic_x(dynamic_X);
    // the general Jacobi path agrees with the fixed-order analytic solution
    EXPECT_LT((dynamic_x - expected).norm(), 1.e-12);
    SymmetricMatrix<double, 2, Cache::Spectral> cached(A);
    const auto retained = cached.evd();
    cached(0, 0) = 4.;
    const auto retained_x = retained.solve(B);
    // retained decomposition factors remain independent after the cached source changes
    EXPECT_LT((retained_x - expected).norm(), 1.e-12);
    const auto refreshed = cached.evd().solve(B);
    const Vector<double, 2> new_expected {2. / 7., -1. / 7.};
    // a new method request uses the refreshed cache and the modified operator
    EXPECT_LT((refreshed - new_expected).norm(), 1.e-12);
    const auto owned = [] {
        const Matrix<double, 2, 2> local_A({2., 1., 1., 2.});
        const Vector<double, 2> local_B {1., 0.};
        return EVD(local_A).solve(local_B);
    }();
    // solution coefficients survive destruction of their source and temporary decomposition
    EXPECT_LT((owned - expected).norm(), 1.e-12);
}

// normalization preserves scalar precision, zero loads and extreme common scales without projection overflow
TEST(EVDSolve, ScalarAndScaleContracts) {
    const Matrix<float, 2, 2> A({0.f, 2.f, 2.f, 0.f});
    const EVD evd(A);
    const Vector<float, 2> B {4.f, 2.f};
    const auto X = evd.solve(B);
    // float input factors return float-valued solutions
    static_assert(std::same_as<std::remove_cvref_t<decltype(X)>, Vector<float, 2>>);
    const Vector<float, 2> expected {1.f, 2.f};
    // an indefinite float system recovers its known coefficients within float precision
    EXPECT_LT((X - expected).norm(), 1.e-5f);
    const Vector<float, 2> zero = Vector<float, 2>::Zero();
    const auto zero_solution = evd.solve(zero);
    // zero loads remain exactly zero without a zero normalization denominator
    EXPECT_EQ(zero_solution, zero);
    const Matrix<double, 2, 2> base({0., 2., 2., 0.});
    const Vector<double, 2> coefficients {.5, -1.};
    for (const double scale : {1.e-150, 1.e150}) {
        const Matrix<double, 2, 2> scaled(scale * base);
        const Vector<double, 2> load(scaled * coefficients);
        const EVD scaled_evd(scaled);
        const auto solution = scaled_evd.solve(load);
        // uniform input scaling preserves the independently specified solution and signed spectrum
        EXPECT_LT((solution - coefficients).norm(), 1.e-12);
    }
    const Matrix<double, 2, 2> large({0., 1.e308, 1.e308, 0.});
    const EVD large_evd(large);
    const Vector<double, 2> load {1.5e308, 1.5e308};
    const auto large_solution = large_evd.solve(load);
    const Vector<double, 2> large_expected {1.5, 1.5};
    // normalized projection avoids overflow of Q-transpose*B while the final solution remains representable
    EXPECT_LT((large_solution - large_expected).norm(), 1.e-12);
}

// unavailable factors, numerical singularity and invalid loads cannot publish a spurious finite solution
TEST(EVDSolve, RejectsInvalidStatesSpectraAndLoads) {
    EVD<Matrix<double, 2, 2>> empty;
    const Vector<double, 2> B {1., 0.};
    // default construction has no eigenpairs available for solving
    EXPECT_THROW(empty.solve(B), std::logic_error);
    const Matrix<double, 2, 2> singular({1., 0., 0., 0.});
    const EVD singular_evd(singular);
    // a zero eigenvalue prevents the unique solution promised by solve
    EXPECT_THROW(singular_evd.solve(B), std::domain_error);
    const Matrix<double, 2, 2> zero = Matrix<double, 2, 2>::Zero();
    const EVD zero_evd(zero);
    // an entirely zero spectrum cannot be normalized or inverted
    EXPECT_THROW(zero_evd.solve(B), std::domain_error);
    const Matrix<double, 2, 2> near_singular({1., 0., 0., -1.e-160});
    const EVD near_evd(near_singular);
    // tiny negative eigenvalues are rejected by magnitude relative to the largest eigenvalue
    EXPECT_THROW(near_evd.solve(B), std::domain_error);
    const Matrix<double, 2, 2> A({2., 1., 1., 2.});
    EVD evd(A);
    const Vector<double, Dynamic> wrong(std::vector<double> {1., 2., 3.});
    // runtime load rows must match the operator order
    EXPECT_THROW(evd.solve(wrong), std::invalid_argument);
    const Matrix<double, 2, Dynamic> no_columns(2, 0);
    // an empty collection of right-hand sides is rejected before projection
    EXPECT_THROW(evd.solve(no_columns), std::invalid_argument);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        const Vector<double, 2> nonfinite {1., bad};
        // every incoming coefficient must be finite before any spectral solve proceeds
        EXPECT_THROW(evd.solve(nonfinite), std::invalid_argument);
    }
    const Matrix<double, 2, 2> asymmetric({2., 3., 1., 2.});
    // failed recomputation removes the completed state used by solve
    EXPECT_THROW(evd.compute(asymmetric), std::invalid_argument);
    // stale eigenpairs cannot be consumed after invalid replacement input
    EXPECT_THROW(evd.solve(B), std::logic_error);
    const double tiny_data[] = {1.e-308};
    const Matrix<double, 1, 1> tiny(tiny_data);
    const EVD tiny_evd(tiny);
    const Vector<double, 1> huge {1.e308};
    // a nonsingular but unrepresentable solution is reported as a numerical error
    EXPECT_THROW(tiny_evd.solve(huge), std::domain_error);
    const Matrix<double, 4, 4> overflowing(
      {1.e308, 1.e308, 0., 0., 1.e308, 1.e308, 0., 0., 0., 0., 1., 0., 0., 0., 0., -1.});
    const EVD overflowing_evd(overflowing);
    const Vector<double, 4> load(1.);
    // nonrepresentable eigenvalues cannot be silently consumed by the solve path
    EXPECT_THROW(overflowing_evd.solve(load), std::domain_error);
}

}   // namespace
