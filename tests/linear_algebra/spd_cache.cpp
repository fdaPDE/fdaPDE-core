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

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace fdapde;

using complete_cache =
  Cache::Union<Cache::Spectral, Cache::Log, Cache::Sqrt, Cache::InverseSqrt, Cache::LogDividedDifferences>;
using uncached_fixed = SPDMatrix<double, 2, 2>;
using cached_fixed = SPDMatrix<double, 2, 2, complete_cache>;
using cached_dynamic = SPDMatrix<double, Dynamic, Dynamic, complete_cache>;

template <typename T>
concept permits_rvalue_view = requires(T value) { std::move(value).view(); };
template <typename T>
concept permits_rvalue_cache = requires(T value) { std::move(value).cache(); };

// the default cache policy stores no additional state for a fixed owner
static_assert(sizeof(uncached_fixed) == sizeof(SymmetricMatrix<double, 2, 2>));
// the default cache policy preserves fixed-owner alignment
static_assert(alignof(uncached_fixed) == alignof(SymmetricMatrix<double, 2, 2>));
// the empty cache also disappears from a dynamic owner layout
static_assert(sizeof(SPDMatrix<double, Dynamic, Dynamic>) == sizeof(SymmetricMatrix<double, Dynamic, Dynamic>));
// the empty cache preserves dynamic-owner alignment
static_assert(alignof(SPDMatrix<double, Dynamic, Dynamic>) == alignof(SymmetricMatrix<double, Dynamic, Dynamic>));
// no-cache fixed views match the native symmetric view's binding layout
static_assert(sizeof(uncached_fixed::View) == sizeof(SymmetricMatrix<double, 2, 2>::View));
// no-cache fixed views preserve the native symmetric view's alignment
static_assert(alignof(uncached_fixed::View) == alignof(SymmetricMatrix<double, 2, 2>::View));
// no-cache dynamic views carry no cache pointer beyond their native symmetric binding
static_assert(
  sizeof(SPDMatrix<double, Dynamic, Dynamic>::ConstView) ==
  sizeof(SymmetricMatrix<double, Dynamic, Dynamic>::ConstView));
// no-cache dynamic views preserve the native symmetric binding's alignment
static_assert(
  alignof(SPDMatrix<double, Dynamic, Dynamic>::ConstView) ==
  alignof(SymmetricMatrix<double, Dynamic, Dynamic>::ConstView));
// a temporary owner cannot leak a dangling mutable or const SPD view
static_assert(!permits_rvalue_view<cached_fixed>);
// a temporary owner cannot leak its cache slot
static_assert(!permits_rvalue_cache<cached_fixed>);

Matrix<double, 2, 2> diagonal(double first, double second) { return Matrix<double, 2, 2>({first, 0, 0, second}); }

template <typename Actual, typename Expected>
void expect_symmetric_near(const Actual& actual, const Expected& expected) {
    // elementwise comparison requires equal row extents before indexing both operands
    ASSERT_EQ(actual.rows(), expected.rows());
    // elementwise comparison requires equal column extents before indexing both operands
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            // each cached coefficient agrees with the independently materialized spectral result
            EXPECT_NEAR(actual(i, j), expected(i, j), 1.0e-12);
        }
    }
}

// default storage has no cache overhead while requested quantities remain selectable at compile time
TEST(SPDCache, DefaultLayoutAndSelectivePolicyContracts) {
    using log_only = SPDMatrix<double, 2, 2, Cache::Log>;
    using log_slot = typename log_only::CacheSlot;
    // the log-only policy records the requested logarithm quantity
    static_assert(log_slot::template Has<Cache::Log>);
    // the log-only policy does not accidentally retain spectral factors
    static_assert(!log_slot::template Has<Cache::Spectral>);
    // the full policy retains each independently selectable quantity
    static_assert(cached_fixed::CacheSlot::template Has<Cache::Spectral>);
    // the full policy retains the logarithm
    static_assert(cached_fixed::CacheSlot::template Has<Cache::Log>);
    // the full policy retains the principal square root
    static_assert(cached_fixed::CacheSlot::template Has<Cache::Sqrt>);
    // the full policy retains the inverse principal square root
    static_assert(cached_fixed::CacheSlot::template Has<Cache::InverseSqrt>);
    // the full policy retains logarithmic divided differences
    static_assert(cached_fixed::CacheSlot::template Has<Cache::LogDividedDifferences>);

    const log_only point(diagonal(4, 9));
    const auto expected = matrix_log(uncached_fixed(diagonal(4, 9)));
    // the selected logarithm is materialized during checked construction
    expect_symmetric_near(point.cache().template matrix<Cache::Log>(), expected);
}

// identity construction prepares all selected static cache quantities without a decomposition
TEST(SPDCache, StaticIdentityPreparesCompleteCache) {
    const auto point = cached_fixed::Identity();
    const auto vectors = point.cache().eigenvectors();
    const auto values = point.cache().eigenvalues();
    const auto logarithm = point.cache().template matrix<Cache::Log>();
    const auto square_root = point.cache().template matrix<Cache::Sqrt>();
    const auto inverse_square_root = point.cache().template matrix<Cache::InverseSqrt>();
    const auto differences = point.cache().log_divided_differences();
    for (int i = 0; i < point.rows(); ++i) {
        // identity eigenvalues remain one in the cached spectral order
        EXPECT_EQ(values[i], 1);
        for (int j = 0; j < point.cols(); ++j) {
            // identity eigenvectors form the canonical basis
            EXPECT_EQ(vectors(i, j), i == j ? 1 : 0);
            // the logarithm of identity is the zero matrix
            EXPECT_EQ(logarithm(i, j), 0);
            // the principal square root of identity is identity
            EXPECT_EQ(square_root(i, j), i == j ? 1 : 0);
            // the inverse principal square root of identity is identity
            EXPECT_EQ(inverse_square_root(i, j), i == j ? 1 : 0);
            // every logarithmic divided difference at identity equals one
            EXPECT_EQ(differences(i, j), 1);
        }
    }
}

// identity construction gives dynamic owners the requested order and fully initialized cache
TEST(SPDCache, DynamicIdentityPreparesCompleteCache) {
    const auto point = cached_dynamic::Identity(3);
    const auto values = point.cache().eigenvalues();
    const auto differences = point.cache().log_divided_differences();
    // the dynamic identity preserves its runtime order
    EXPECT_EQ(point.rows(), 3);
    // the cache uses the same runtime order as its owner
    EXPECT_EQ(point.cache().rows(), 3);
    for (int i = 0; i < point.rows(); ++i) {
        // every dynamic identity eigenvalue is one
        EXPECT_EQ(values[i], 1);
        for (int j = 0; j < point.cols(); ++j) {
            // the dynamic identity has canonical coefficients
            EXPECT_EQ(point(i, j), i == j ? 1 : 0);
            // the repeated identity spectrum uses the exact divided-difference limit
            EXPECT_EQ(differences(i, j), 1);
        }
    }
}

// copies and mixed-policy materializations own independent caches while preserving common quantities
TEST(SPDCache, CopiesAndMixedPoliciesPreserveIndependentQuantities) {
    using log_only = SPDMatrix<double, 2, 2, Cache::Log>;
    const log_only source(diagonal(4, 9));
    cached_fixed complete(source);
    const log_only reduced(complete);
    cached_fixed assigned(diagonal(16, 25));
    assigned.assign(source);
    const cached_fixed copy(complete);
    const SPDMatrix<double, 2, 2, Cache::Union<Cache::Spectral, Cache::Log>> spectral_log(complete);
    const cached_fixed expanded(spectral_log);

    // a broader policy reconstructs the missing spectral quantities from the verified value
    EXPECT_NEAR(complete.cache().eigenvalues()[0] * complete.cache().eigenvalues()[1], 36, 1.0e-12);
    // the common logarithm is reused when a broader policy is materialized
    expect_symmetric_near(complete.cache().template matrix<Cache::Log>(), source.cache().template matrix<Cache::Log>());
    // a narrower policy retains the copied logarithm
    expect_symmetric_near(reduced.cache().template matrix<Cache::Log>(), source.cache().template matrix<Cache::Log>());
    // assignment from a different policy refreshes selected quantities from the verified source
    expect_symmetric_near(assigned.cache().template matrix<Cache::Log>(), source.cache().template matrix<Cache::Log>());
    // copied owners do not share their cache buffers
    EXPECT_NE(copy.cache().data(), complete.cache().data());
    for (std::size_t i = 0; i < cached_fixed::CacheSlot::scalar_count(2); ++i) {
        // policy expansion preserves copied factors and reconstructs missing quantities from the same spectrum
        EXPECT_NEAR(expanded.cache().data()[i], complete.cache().data()[i], 1e-12);
    }

    const auto copy_logarithm = matrix_log(copy);
    complete.assign(diagonal(16, 25));
    // replacing one owner cannot mutate a copied owner's cache
    expect_symmetric_near(matrix_log(copy), copy_logarithm);
    // the replaced owner now has the logarithm of its new coefficients
    expect_symmetric_near(complete.cache().template matrix<Cache::Log>(), matrix_log(uncached_fixed(diagonal(16, 25))));
}

// policy expansion skips reusable quantities and rebuilds divided differences only when pairing a new spectral basis
TEST(SPDCache, CrossPolicyCopyKeepsSpectralAndDividedDifferenceBasesCoherent) {
    using spectral_and_differences = Cache::Union<Cache::Spectral, Cache::LogDividedDifferences>;
    using spectral_point = SPDMatrix<double, 2, 2, spectral_and_differences>;
    using differences_only_point = SPDMatrix<double, 2, 2, Cache::LogDividedDifferences>;

    const Matrix<double, 2, 2> coefficients({4.0, 1.0, 1.0, 3.0});
    const spectral_point source(coefficients);
    const differences_only_point without_basis(source);
    const spectral_point restored(without_basis);
    const uncached_fixed uncached(coefficients);
    SymmetricMatrix<double, 2, 2> direction;
    direction(0, 0) = 0.5;
    direction(1, 0) = -0.7;
    direction(1, 1) = 1.2;

    // the restored cache evaluates the noncommuting derivative in the same basis as an independent decomposition
    expect_symmetric_near(matrix_log_frechet(restored, direction), matrix_log_frechet(uncached, direction));

    using differences_and_log = Cache::Union<Cache::LogDividedDifferences, Cache::Log>;
    using slot_type = internals::spd_cache_slot<double, 2, differences_and_log>;
    std::vector<double> storage(slot_type::scalar_count(2), -7.0);
    slot_type slot(storage.data(), 2);
    const EVD<SymmetricMatrix<double, 2, 2>> spectral(uncached.rep());
    slot.prepare<Cache::LogDividedDifferences>(spectral);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            // missing-only preparation leaves each retained divided-difference sentinel untouched
            EXPECT_EQ(slot.log_divided_differences()(i, j), -7.0);
        }
    // the missing logarithm is still reconstructed from the supplied decomposition
    expect_symmetric_near(slot.template matrix<Cache::Log>(), matrix_log(uncached));
    const SPDMatrix<double, 2, 2, differences_and_log> expanded(without_basis);
    // expansion without a retained spectral basis preserves the source divided-difference table
    expect_symmetric_near(expanded.cache().log_divided_differences(), without_basis.cache().log_divided_differences());
}

// cache-backed spectral primitives agree with uncached paths at a repeated spectrum
TEST(SPDCache, CachedPrimitivesMatchUncachedRepeatedSpectrum) {
    const cached_fixed cached(diagonal(2, 2));
    const uncached_fixed uncached(diagonal(2, 2));
    SymmetricMatrix<double, 2, 2> direction;
    direction(0, 0) = 1;
    direction(1, 0) = -0.5;
    direction(1, 1) = 2;
    const auto cached_logarithm = matrix_log(cached);
    const auto uncached_logarithm = matrix_log(uncached);
    const auto cached_derivative = matrix_log_frechet(cached, direction);
    const auto uncached_derivative = matrix_log_frechet(uncached, direction);
    const auto restored = matrix_exp<complete_cache>(cached_logarithm);

    // the repeated-eigenvalue cache stores the analytic logarithmic divided-difference limit
    EXPECT_NEAR(cached.cache().log_divided_differences()(0, 1), 0.5, 1.0e-12);
    // the cached logarithm matches the uncached spectral primitive
    expect_symmetric_near(cached_logarithm, uncached_logarithm);
    // the cached divided differences preserve the Frechet derivative at repeated eigenvalues
    expect_symmetric_near(cached_derivative, uncached_derivative);
    // the policy-bearing exponential restores the original checked SPD value
    expect_symmetric_near(restored, cached);
    // the exponential result initializes the requested spectral cache
    EXPECT_NEAR(restored.cache().eigenvalues()[0] * restored.cache().eigenvalues()[1], 4, 1.0e-12);
}

// views expose read-only values and transfer replacements without changing their binding
TEST(SPDCache, ViewsPreserveReadOnlyAccessAndValueAssignment) {
    cached_fixed owner(diagonal(4, 9));
    auto view = owner.view();
    const auto* const binding = view.data();
    const auto read_only = std::as_const(owner).view();
    const Matrix<double, 2, 2> replacement = diagonal(16, 25);

    // an SPD view always exposes a const scalar pointer
    static_assert(std::is_same_v<decltype(view.data()), const double*>);
    // a const owner produces a const scalar pointer through its view
    static_assert(std::is_same_v<decltype(read_only.data()), const double*>);
    // views preserve the checked SPD read-only expression contract
    static_assert(decltype(view)::ReadOnly == 1);
    // assignment validates and copies a replacement value without rebinding the view
    view.assign(replacement);
    // the view still refers to its original packed owner storage
    EXPECT_EQ(view.data(), binding);
    // value assignment updates the owner through the existing view binding
    EXPECT_EQ(owner(0, 0), 16);
    // value assignment also refreshes the selected cache quantities
    EXPECT_NEAR(view.cache().template matrix<Cache::Log>()(1, 1), std::log(25), 1.0e-12);
}

// ordinary matrix combinations remain unchecked expressions and failed replacements preserve coefficients and cache
TEST(SPDCache, FailedAssignmentProvidesOwnerAndViewStrongGuarantees) {
    cached_fixed owner(diagonal(4, 9));
    const uncached_fixed other(diagonal(16, 25));
    const auto combination = 2.0 * owner + (-1.0) * other;
    // scalar products and sums stay in the native matrix-expression family
    static_assert(std::derived_from<decltype(combination), MatrixExpr<std::remove_cvref_t<decltype(combination)>>>);
    // an arbitrary weighted sum of SPD owners is not itself a verified SPD value
    static_assert(!SPDLike<decltype(combination)>);
    // ordinary arithmetic does not acquire the broader SPD expression tag either
    static_assert(!is_spd_matrix_v<decltype(combination)>);
    const Matrix<double, 2, 2> dense_combination(combination);
    // ordinary dense materialization preserves negative diagonal entries without imposing positivity
    expect_symmetric_near(dense_combination, diagonal(-8, -7));
    // SPD materialization checks and rejects the same nonpositive matrix expression
    EXPECT_THROW((cached_fixed(combination)), std::domain_error);
    auto view = owner.view();
    const auto saved_owner = owner;
    const std::vector<double> saved_cache(
      owner.cache().data(), owner.cache().data() + cached_fixed::CacheSlot::scalar_count(owner.rows()));
    const Matrix<double, 2, 2> indefinite = diagonal(-1, 1);

    // owner assignment rejects a finite symmetric matrix that is not positive definite
    EXPECT_THROW(owner.assign(indefinite), std::domain_error);
    // owner coefficients remain equal to the verified pre-assignment value
    expect_symmetric_near(owner, saved_owner);
    for (std::size_t i = 0; i < saved_cache.size(); ++i) {
        // every selected cache scalar remains unchanged after owner assignment fails
        EXPECT_EQ(owner.cache().data()[i], saved_cache[i]);
    }
    // view assignment applies the same validation before touching bound storage
    EXPECT_THROW(view.assign(indefinite), std::domain_error);
    // view coefficients still observe the unchanged owner value
    expect_symmetric_near(view, saved_owner);
    for (std::size_t i = 0; i < saved_cache.size(); ++i) {
        // every selected cache scalar remains unchanged after view assignment fails
        EXPECT_EQ(view.cache().data()[i], saved_cache[i]);
    }
}

// move and swap keep each owned coefficient set associated with independently valid cache data
TEST(SPDCache, MoveAndSwapPreserveCoefficientCacheAssociation) {
    cached_dynamic first(diagonal(4, 9));
    auto moved = std::move(first);
    // the moved destination retains the spectral determinant of the source value
    EXPECT_NEAR(moved.determinant(), 36, 1e-12);
    auto second = cached_dynamic::Identity(3);
    std::swap(moved, second);
    // swapping runtime-sized owners carries the three-dimensional identity shape with its coefficients
    EXPECT_EQ(moved.rows(), 3);
    // the identity's cache retains its corresponding dimension after swap
    EXPECT_EQ(moved.cache().rows(), 3);
    // the exchanged two-dimensional value retains the source matrix logarithm
    EXPECT_NEAR(second.cache().template matrix<Cache::Log>()(0, 0), std::log(4.), 1e-12);
    // the exchanged owners do not share writable cache buffers
    EXPECT_NE(moved.cache().data(), second.cache().data());
}

}   // namespace
