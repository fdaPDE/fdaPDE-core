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

using fixed_symmetric = SymmetricMatrix<double, 3, 3>;
using fixed_evd = EVD<fixed_symmetric>;
using const_view_evd = EVD<SymmetricMatrixView<const double, 3, 3>>;
// an eigendecomposition may be constructed before receiving a matrix
static_assert(std::is_default_constructible_v<fixed_evd>);
// a const input view still produces owned double-valued factors
static_assert(std::is_same_v<typename const_view_evd::Scalar, double>);
// eigenvalue references cannot escape a temporary decomposition
static_assert(!permits_rvalue_eigenvalues<fixed_evd>);
// an eigenvector adaptor cannot borrow a temporary decomposition
static_assert(!permits_rvalue_eigenvectors<fixed_evd>);

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
    SymmetricMatrix<double, Dynamic, Dynamic> repeated(4, 4);
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
        const SymmetricMatrix<double, 2, 2> matrix({2.0 * scale, scale, 2.0 * scale});
        const auto decomposition = matrix.evd();
        const double minimum = std::min(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]);
        const double maximum = std::max(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]);
        // uniform scaling preserves the smaller normalized eigenvalue one
        EXPECT_NEAR(minimum / scale, 1.0, 1.0e-12);
        // uniform scaling preserves the larger normalized eigenvalue three
        EXPECT_NEAR(maximum / scale, 3.0, 1.0e-12);
        expect_valid_evd(matrix, decomposition);
    }

    const SymmetricMatrix<double, 3, 3> zero({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
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
    EVD<SymmetricMatrix<double, Dynamic, Dynamic>> reusable;
    // a default object has no completed eigendecomposition
    EXPECT_FALSE(reusable.computed());

    SymmetricMatrix<double, Dynamic, Dynamic> valid(2, 2);
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

    const SymmetricMatrix<double, Dynamic, Dynamic> empty;
    // an empty symmetric matrix is rejected
    EXPECT_THROW(reusable.compute(empty), std::invalid_argument);
    // empty input leaves no completed eigendecomposition
    EXPECT_FALSE(reusable.computed());

    EVD<SymmetricMatrix<double, 3, 3>> fixed_shape;
    SymmetricMatrix<double, Dynamic, Dynamic> wrong_shape(2, 2);
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
