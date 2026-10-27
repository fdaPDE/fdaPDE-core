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

using Fixed = Cholesky<Matrix<double, 3, 3>>;
template <typename T>
concept permits_temporary_factor = requires(T value) { std::move(value).L(); };
// factor references cannot escape a temporary decomposition
static_assert(!permits_temporary_factor<Fixed>);
// the lower factor carries the statically known matrix order
static_assert(std::same_as<decltype(std::declval<const Fixed&>().L()), const LowerTriangularMatrix<double, 3>&>);

// dense views of either storage order reconstruct a supplied lower factor and solve multiple right-hand sides
template <int StorageOrder> void check_dense_view() {
    const Matrix<double, 3, 3> expected({2., 0., 0., 1., 3., 0., -.5, .25, 4.});
    const Matrix<double, 3, 3, StorageOrder> A(expected * expected.transpose());
    const MatrixView<const double, 3, 3, StorageOrder> view(A.data());
    const Cholesky chol(view);
    // factor coefficients recover the independently supplied positive-diagonal lower triangle
    EXPECT_LT((chol.L() - expected).norm(), 1.e-12);
    const Matrix<double, 3, 2> B({1., 2., 3., 4., 5., 6.});
    const auto X = chol.solve(B);
    // the solve returns the known factor order and the right-hand-side column count
    static_assert(std::same_as<std::remove_cvref_t<decltype(X)>, Matrix<double, 3, 2>>);
    // residuals against the original dense system validate both substitution passes
    EXPECT_LT((A * X - B).norm(), 1.e-12);
}

// fixed dense views support row-major and column-major storage with independent factor ownership
TEST(Cholesky, DenseViewsAndMatrixSolve) {
    check_dense_view<RowMajor>();
    check_dense_view<ColMajor>();
}

// partial static orders and fully dynamic one-column right-hand sides retain their appropriate shapes
TEST(Cholesky, DynamicShapesAndRecomputation) {
    const Matrix<double, Dynamic, 2> partial(Matrix<double, 2, 2>({4., 2., 2., 5.}));
    const Cholesky fixed(partial);
    // a known column order preserves fixed factor metadata even when rows are dynamic
    static_assert(std::same_as<decltype(fixed.L()), const LowerTriangularMatrix<double, 2>&>);
    Matrix<double, Dynamic, Dynamic> A(partial);
    Cholesky chol(A);
    const Matrix<double, Dynamic, Dynamic> B(Vector<double, 2> {1., 3.});
    const auto X = chol.solve(B);
    // a runtime one-column matrix is a valid right-hand side despite its dynamic column metadata
    EXPECT_LT((A * X - B).norm(), 1.e-12);
    A.resize(1, 1);
    A(0, 0) = 9.;
    chol.compute(A);
    // recomputation may change the order of a fully dynamic decomposition
    EXPECT_EQ(chol.L().rows(), 1);
    // the one-dimensional solve agrees with direct division by the new coefficient
    EXPECT_NEAR(chol.solve(Vector<double, 1> {18.})[0], 2., 1.e-14);
}

// finite extreme-scale matrices use extended pivot accumulation without overflow or underflow
TEST(Cholesky, ScaledPositiveInputs) {
    const Matrix<double, 2, 2> unscaled({4., 2., 2., 5.});
    for (double scale : {1.e-200, 1.e200}) {
        const Matrix<double, 2, 2> A(scale * unscaled);
        const Cholesky chol(A);
        const Matrix<double, 2, 2> reconstructed(chol.L() * chol.L().transpose());
        // normalization compares the reconstructed factor product independently of the input magnitude
        EXPECT_LT((reconstructed / scale - A / scale).norm(), 1.e-12);
    }
}

// rejected matrix and right-hand-side inputs cannot expose an incomplete decomposition
TEST(Cholesky, InvalidInputsAndUnavailableFactors) {
    Fixed empty;
    const Fixed copied(empty);
    // copying an uncomputed decomposition preserves the absence of factor storage
    EXPECT_THROW(copied.L(), std::logic_error);
    // default construction cannot publish a factor before compute succeeds
    EXPECT_THROW(empty.L(), std::logic_error);
    // an uncomputed decomposition cannot solve a system
    EXPECT_THROW(empty.solve(Vector<double, 3> {1., 2., 3.}), std::logic_error);
    const Matrix<double, 2, 2> A({4., 2., 2., 5.});
    Cholesky chol(A);
    // asymmetry is rejected rather than silently selecting one triangle
    EXPECT_THROW(chol.compute(Matrix<double, 2, 2>({4., 3., 2., 5.})), std::invalid_argument);
    // failed recomputation hides the previous factor
    EXPECT_FALSE(chol.computed());
    // the factor getter rejects stale results after failure
    EXPECT_THROW(chol.L(), std::logic_error);
    // a symmetric indefinite input fails the positive-pivot requirement
    EXPECT_THROW(chol.compute(Matrix<double, 2, 2>({1., 2., 2., 1.})), std::domain_error);
    // semidefinite inputs cannot supply the strictly positive pivots needed for a solve
    EXPECT_THROW(chol.compute(Matrix<double, 2, 2>({1., 1., 1., 1.})), std::domain_error);
    // nonfinite input coefficients fail before a factor is published
    EXPECT_THROW(
      chol.compute(Matrix<double, 2, 2>({1., 0., 0., std::numeric_limits<double>::infinity()})), std::invalid_argument);
    Cholesky<Matrix<double, Dynamic, Dynamic>> dynamic;
    // generic rectangular matrices cannot be factored by Cholesky
    EXPECT_THROW(dynamic.compute(Matrix<double, Dynamic, Dynamic>(2, 3)), std::invalid_argument);
    // empty matrices cannot publish a factor
    EXPECT_THROW(dynamic.compute(Matrix<double, Dynamic, Dynamic>(0, 0)), std::invalid_argument);
    chol.compute(A);
    // right-hand sides must have the same row count as the factor
    EXPECT_THROW(chol.solve(Vector<double, 3> {1., 2., 3.}), std::invalid_argument);
    // nonfinite right-hand sides cannot produce a valid solution
    EXPECT_THROW(chol.solve(Vector<double, 2> {1., std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
}

// SPD methods work with no cache, an unrelated cache, and a retained lower factor on owners and views
TEST(Cholesky, SPDMethodsAndCachedFactors) {
    using Policy = Cache::Union<Cache::Cholesky, Cache::LogCholesky>;
    const Matrix<double, 2, 2> A({4., 2., 2., 5.});
    const SPDMatrix<double, 2> plain(A);
    const SPDMatrix<double, 2, Cache::Spectral> spectral(A);
    SPDMatrix<double, 2, Policy> cached(A);
    const auto original = cached.cholesky();
    const auto temporary = SPDMatrix<double, 2>(A).cholesky();
    // an owning decomposition remains valid after its temporary source is destroyed
    EXPECT_LT((temporary.L() * temporary.L().transpose() - A).norm(), 1.e-14);
    const auto viewed = std::as_const(cached).view().cholesky();
    const auto retained = cached.cache().cholesky();
    // retained lower factors now carry triangular structure through a packed const view
    static_assert(std::same_as<std::remove_cvref_t<decltype(retained)>, LowerTriangularMatrixView<const double, 2>>);
    // the factor buffer retains only the three independent lower-triangle coefficients
    EXPECT_EQ((SPDMatrix<double, 2, Cache::Cholesky>::CacheSlot::scalar_count(2)), 3u);
    // a no-cache owner computes the same unique positive-diagonal factor
    const auto plain_chol = plain.cholesky();
    EXPECT_LT((plain_chol.L() - original.L()).norm(), 1.e-14);
    // selecting spectral storage does not make Cholesky cache selection mandatory
    const auto spectral_chol = spectral.cholesky();
    EXPECT_LT((spectral_chol.L() - original.L()).norm(), 1.e-14);
    // cached views return the same coefficients as their owner-backed factor
    EXPECT_LT((viewed.L() - original.L()).norm(), 1.e-14);
    // the decomposition owns independent packed coefficients rather than borrowing the input factor
    EXPECT_NE(original.L().data(), retained.data());
    // copying retained coefficients preserves their exact scalar values without a new factorization
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j <= i; ++j) EXPECT_EQ(original.L()(i, j), retained(i, j));
    cached = SPDMatrix<double, 2, Policy>::Identity();
    // replacing the source does not alter a previously returned owning decomposition
    EXPECT_LT((original.L() * original.L().transpose() - A).norm(), 1.e-14);
    const auto replacement = cached.cholesky();
    const Matrix<double, 2, 2> identity({1., 0., 0., 1.});
    // subsequent method requests use the newly prepared identity factor
    EXPECT_LT((replacement.L() - identity).norm(), 1.e-14);
}
// direct SPD factor requests preserve shape, reuse cached coefficients and own results independently of the source
TEST(Cholesky, DirectSPDFactors) {
    const Matrix<double, 2, 2> A({4., 2., 2., 5.});
    const Matrix<double, 2, 2> expected({2., 0., 1., 2.});
    const SPDMatrix<double, 2> plain(A);
    const SPDMatrix<double, 2, Cache::Spectral> spectral(A);
    SPDMatrix<double, 2, Cache::Cholesky> cached(A);
    const auto factor = cached.cholesky_factor();
    const auto retained = cached.cache().cholesky();
    // the direct getter returns a structured owner with the input's fixed scalar type and order
    static_assert(std::same_as<std::remove_cvref_t<decltype(factor)>, LowerTriangularMatrix<double, 2>>);
    // the returned packed buffer is independent of the retained cache buffer
    EXPECT_NE(factor.data(), retained.data());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j <= i; ++j)
            // cached requests preserve each retained coefficient exactly without refactoring
            EXPECT_EQ(factor(i, j), retained(i, j));
    // the cached factor agrees with the independently supplied positive-diagonal lower triangle
    EXPECT_LT((factor - expected).norm(), 1.e-14);
    const auto plain_factor = plain.cholesky_factor();
    // selecting no cache still computes the unique factor of the original coefficients
    EXPECT_LT((plain_factor - expected).norm(), 1.e-14);
    const auto spectral_factor = spectral.cholesky_factor();
    // an unrelated cache policy does not restrict access to the Cholesky factor
    EXPECT_LT((spectral_factor - expected).norm(), 1.e-14);
    const auto plain_view_factor = plain.view().cholesky_factor();
    // a const no-cache view supports the same direct factor request as its owner
    EXPECT_LT((plain_view_factor - expected).norm(), 1.e-14);
    const auto viewed = std::as_const(cached).view().cholesky_factor();
    const auto temporary = SPDMatrix<double, 2, Cache::Cholesky>(A).cholesky_factor();
    // the owning factor remains available after a temporary source has been destroyed
    EXPECT_LT((temporary - expected).norm(), 1.e-14);
    cached = SPDMatrix<double, 2, Cache::Cholesky>::Identity();
    // source replacement leaves previously returned owner factors unchanged
    EXPECT_LT((factor * factor.transpose() - A).norm(), 1.e-14);
    // factors requested through a view also own coefficients independently of later source replacement
    EXPECT_LT((viewed - expected).norm(), 1.e-14);
    const Matrix<double, 2, 2> identity({1., 0., 0., 1.});
    const auto replacement = cached.cholesky_factor();
    // a subsequent getter copies the refreshed identity factor rather than the old cached coefficients
    EXPECT_LT((replacement - identity).norm(), 1.e-14);
    const SPDMatrix<double, Dynamic> dynamic(A);
    const auto dynamic_factor = dynamic.cholesky_factor();
    // dynamic source orders remain dynamic in the returned structured owner
    static_assert(std::same_as<std::remove_cvref_t<decltype(dynamic_factor)>, LowerTriangularMatrix<double, Dynamic>>);
    // the dynamic result reconstructs the original input using its runtime order
    EXPECT_LT((dynamic_factor * dynamic_factor.transpose() - A).norm(), 1.e-14);
}
}   // namespace
