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

namespace {
using namespace fdapde;
using Dense = Matrix<double, 2, 2>;
using Cached = CachedSymmetricMatrix<double, 2, 2>;

/// @brief supplies unrelated eigenpairs to check that arbitrary expressions cannot forge a native cache contract
struct spoofed_symmetric : SymmetricMatrixExpr<spoofed_symmetric> {
    using Scalar = double;
    static constexpr int Rows = 2, Cols = 2;
    /// @brief returns the fixed row count
    int rows() const { return 2; }
    /// @brief returns the fixed column count
    int cols() const { return 2; }
    /// @brief represents an indefinite diagonal matrix independently of the supplied cache
    double operator()(int i, int j) const { return i == j ? (i == 0 ? -1. : 2.) : 0.; }
    /// @brief deliberately supplies positive identity eigenpairs unrelated to the coefficients
    const auto& cache() const { return identity.cache(); }
    SPDMatrix<double, 2, 2, Cache::Spectral> identity = SPDMatrix<double, 2, 2, Cache::Spectral>::Identity();
};

// only controlled native types can supply reusable eigenpairs to EVD and matrix-function domain checks
TEST(SymmetricCache, RejectsUntrustedSpectralMetadata) {
    const spoofed_symmetric source;
    const EVD decomposition(source);
    // the native solver must recover the negative coefficient instead of copying the unrelated positive cache
    EXPECT_LT(std::min(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]), 0);
    // a cache-shaped user expression cannot bypass the real logarithm's positivity check
    EXPECT_THROW(logm(source), std::domain_error);
}

// writes through owners, views and retained proxies invalidate the same per-matrix spectral state
TEST(SymmetricCache, MutationAndRetainedAliases) {
    Cached matrix(Dense({2, .3, .3, 4}));
    auto alias = matrix(0, 0);
    auto view = matrix.view();
    Cached::ConstView read(view);
    const auto& slot = matrix.cache();
    // explicit cache access prepares eigenpairs for the current coefficients
    EXPECT_TRUE(slot.valid());
    alias = 5;
    // a coefficient proxy captured before preparation still invalidates on a later write
    EXPECT_FALSE(slot.valid());
    // a retained cache slot explicitly rejects stale eigenpair access after invalidation
    EXPECT_THROW(slot.eigenvalues(), std::logic_error);
    const auto actual = expm(read);
    const auto expected = expm(Dense({5, .3, .3, 4}));
    // matrix functions through a const alias see the new coefficients and refreshed eigenpairs
    EXPECT_NEAR(actual(0, 0), expected(0, 0), 1e-11);
    view(0, 1) = .8;
    // mutation through the view invalidates the same owner cache
    EXPECT_FALSE(slot.valid());
    // packed symmetric writes update the reflected coefficient as well
    EXPECT_DOUBLE_EQ(double(matrix(1, 0)), .8);
    matrix = Dense({6, .4, .4, 4});
    // same-shape owner replacement preserves a saved proxy and its shared invalidation slot
    alias = 8;
    EXPECT_DOUBLE_EQ(read(0, 0), 8);
    const auto independent = matrix;
    matrix(1, 1) = 7;
    // an owning copy keeps independent coefficients when the original is mutated
    EXPECT_DOUBLE_EQ(independent(1, 1), 4);
    // cached owners expose no raw mutable coefficient alias
    static_assert(std::same_as<decltype(matrix.data()), const double*>);
}

// spectral caching preserves general symmetric domains rather than silently assuming positive definiteness
TEST(SymmetricCache, DomainsAndBatchInvalidation) {
    Cached indefinite(Dense({-1, 0, 0, 2}));
    // a general symmetric cache retains negative eigenvalues without rejecting its owner
    EXPECT_NO_THROW(indefinite.cache());
    // a restricted-domain matrix logarithm checks positivity even when eigenpairs are cached
    EXPECT_THROW(logm(indefinite), std::domain_error);
    const auto exponential = matrix_exp(indefinite);
    // the exponential of the cached indefinite diagonal matrix matches its scalar formula
    EXPECT_NEAR(exponential(0, 0), std::exp(-1.), 1e-13);
    MatrixBatch<Cached> batch(2);
    batch[0] = Dense({2, .2, .2, 3});
    const auto copy = batch;
    auto view = batch[0];
    auto alias = view(0, 0);
    const auto& slot = view.cache();
    alias = 4;
    // a saved batch coefficient alias invalidates its corresponding aggregate cache slot
    EXPECT_FALSE(slot.valid());
    const auto actual = sqrtm(batch[0]);
    const auto expected = sqrtm(Dense({4, .2, .2, 3}));
    // spectral functions refresh the modified batch slot before evaluating restricted-domain operations
    EXPECT_NEAR(actual(0, 0), expected(0, 0), 1e-12);
    // copying a batch gives independent coefficient buffers and cache slot bindings
    EXPECT_DOUBLE_EQ(copy[0](0, 0), 2);
    const double huge = .9 * std::numeric_limits<double>::max();
    // symmetry validation stays meaningful when a finite matrix has an overflowing Frobenius norm
    EXPECT_THROW((Cached(Dense({huge, .1 * huge, .3 * huge, huge}))), std::invalid_argument);
    const CachedSymmetricMatrix<double, 2, 2, Cache::None> uncached(Dense({4, .2, .2, 3}));
    // disabling the cache preserves the same matrix-function values
    EXPECT_NEAR(expm(uncached)(0, 0), expm(batch[0])(0, 0), 1e-11);
}
// dynamic cache wrappers preserve runtime shape and scalar precision through mutation and spectral reuse
TEST(SymmetricCache, DynamicFloatAndShapeContracts) {
    using DynamicCached = CachedSymmetricMatrix<float, Dynamic, Dynamic, Cache::Union<Cache::Spectral, Cache::None>>;
    DynamicCached value(Matrix<float, 3, 3>({2, 0, 0, 0, 3, 0, 0, 0, 4}));
    const auto exponential = matrix_exp(value);
    // scalar-preserving checked exponentials agree with the analytic float diagonal
    EXPECT_NEAR(exponential(2, 2), std::exp(4.f), 2e-5);
    MatrixBatch<DynamicCached> points(2, 3, 3);
    points[0] = value;
    points[0](0, 0) = 5;
    // a runtime-shaped batch view refreshes the eigenpairs after its coefficient update
    EXPECT_NEAR(matrix_exp(points[0])(0, 0), std::exp(5.f), 1e-4);
    // a fixed-shape cached owner rejects incompatible dynamic source dimensions at its public boundary
    EXPECT_THROW(
      (Cached(Matrix<double, Dynamic, Dynamic>(IdentityMatrix<double, Dynamic, Dynamic>(3, 3)))),
      std::invalid_argument);
}

}   // namespace
