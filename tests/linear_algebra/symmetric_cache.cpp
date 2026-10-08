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
using Cached = SymmetricMatrix<double, 2, Cache::Spectral>;

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
    SPDMatrix<double, 2, Cache::Spectral> identity = SPDMatrix<double, 2, Cache::Spectral>::Identity();
};

// only controlled native types can supply reusable eigenpairs to EVD and matrix-function domain checks
TEST(SymmetricCache, RejectsUntrustedSpectralMetadata) {
    const spoofed_symmetric source;
    const EVD decomposition(source);
    // the native solver must recover the negative coefficient instead of copying the unrelated positive cache
    EXPECT_LT(std::min(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]), 0);
    const auto values = source.eigenvalues();
    // direct eigenvalue extraction also rejects forged metadata and recovers the negative diagonal entry
    EXPECT_NEAR(std::min(values[0], values[1]), -1., 1e-13);
    // a cache-shaped user expression cannot bypass the real logarithm's positivity check
    EXPECT_THROW(logm(source), std::domain_error);
    const auto exponential = source.exp();
    // direct exponentiation ignores the forged positive spectrum and exponentiates the actual negative entry
    EXPECT_NEAR(exponential(0, 0), std::exp(-1.), 1e-13);
    const auto inverse = source.inv();
    // symmetric inversion ignores forged factors and retains the negative reciprocal from actual coefficients
    EXPECT_NEAR(inverse(0, 0), -1., 1e-13);
}

// direct eigenvalue extraction owns its result and refreshes tracked writes through a retained const view
TEST(SymmetricCache, EigenvaluesOwnershipAndTrackedWrites) {
    Cached matrix(Vector<double, 3> {2., 0., 3.});
    const auto view = std::as_const(matrix).view();
    const auto values = view.eigenvalues();
    // a const-scalar view returns a mutable-scalar owning vector with its fixed order
    static_assert(std::same_as<std::remove_cv_t<decltype(values)>, Vector<double, 2>>);
    const auto& slot = matrix.cache();
    matrix(0, 0) = 5.;
    // the tracked diagonal write invalidates the factors retained before the mutation
    EXPECT_FALSE(slot.valid());
    const auto changed = view.eigenvalues();
    // direct extraction refreshes the same owner slot rather than decomposing into unrelated temporary factors
    EXPECT_TRUE(slot.valid());
    // the retained const view refreshes its owner's changed diagonal, whose trace is five plus three
    EXPECT_NEAR(changed.sum(), 8., 1e-13);
    // the previously returned vector retains the original two-plus-three trace after owner mutation
    EXPECT_NEAR(values.sum(), 5., 1e-13);
    const auto temporary_values = Cached(Vector<double, 3> {4., 0., 5.}).eigenvalues();
    // an eigenvalue vector outlives a temporary matrix and retains its four-plus-five trace
    EXPECT_NEAR(temporary_values.sum(), 9., 1e-13);
}

// symmetric sources and SPD cache policies expose the same owning eigenvalue API
TEST(SymmetricCache, EigenvaluesUncachedAndSPD) {
    const Vector<double, 3> packed {2., 1., 2.};
    const SymmetricMatrix<double, 2> symmetric(packed);
    const auto values = symmetric.eigenvalues();
    // the uncached symmetric matrix has the analytic spectrum one and three independently of ordering
    EXPECT_NEAR(std::min(values[0], values[1]), 1., 1e-13);
    // the second analytic eigenvalue follows from the equal diagonal and unit off-diagonal
    EXPECT_NEAR(std::max(values[0], values[1]), 3., 1e-13);
    const SPDMatrix<double, 2> plain(packed);
    const auto plain_values = plain.eigenvalues();
    // uncached SPD dispatch preserves the analytic determinant two times two minus one
    EXPECT_NEAR(plain_values[0] * plain_values[1], 3., 1e-13);
    const SPDMatrix<double, 2, Cache::Spectral> cached(packed);
    const auto cached_values = cached.eigenvalues();
    // cached spectral factors expose the same SPD determinant without explicit cache access
    EXPECT_NEAR(cached_values[0] * cached_values[1], 3., 1e-13);
    const SPDMatrix<double, 2, Cache::Log> logarithmic(packed);
    const auto logarithmic_values = logarithmic.eigenvalues();
    // the log-only cache lacks eigenpairs, so fallback decomposition must recover the same determinant
    EXPECT_NEAR(logarithmic_values[0] * logarithmic_values[1], 3., 1e-13);
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
    // mutable owners expose writable packed coefficients through the invalidating overload
    static_assert(std::same_as<decltype(matrix.data()), double*>);
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
    const SymmetricMatrix<double, 2, Cache::None> uncached(Dense({4, .2, .2, 3}));
    // disabling the cache preserves the same matrix-function values
    EXPECT_NEAR(expm(uncached)(0, 0), expm(batch[0])(0, 0), 1e-11);
}
// dynamic cache wrappers preserve runtime shape and scalar precision through mutation and spectral reuse
TEST(SymmetricCache, DynamicFloatAndShapeContracts) {
    using DynamicCached = SymmetricMatrix<float, Dynamic, Cache::Union<Cache::Spectral, Cache::None>>;
    DynamicCached value(Matrix<float, 3, 3>({2, 0, 0, 0, 3, 0, 0, 0, 4}));
    const auto exponential = matrix_exp(value);
    // scalar-preserving checked exponentials agree with the analytic float diagonal
    EXPECT_NEAR(exponential(2, 2), std::exp(4.f), 2e-5);
    MatrixBatch<DynamicCached> points(2, 3, 3);
    points[0] = value;
    points[0](0, 0) = 5;
    const auto values = std::as_const(points)[0].eigenvalues();
    // a runtime-shaped const batch view returns an owning float vector instead of promoting its scalar type
    static_assert(std::same_as<std::remove_cv_t<decltype(values)>, Vector<float, Dynamic>>);
    // the returned vector uses the runtime order of the source matrix
    EXPECT_EQ(values.size(), 3);
    // direct eigenvalue extraction observes the updated five-plus-three-plus-four diagonal trace
    EXPECT_NEAR(values.sum(), 12.f, 1e-5f);
    // a runtime-shaped batch view refreshes the eigenpairs after its coefficient update
    EXPECT_NEAR(matrix_exp(points[0])(0, 0), std::exp(5.f), 1e-4);
    // a fixed-shape cached owner rejects incompatible dynamic source dimensions at its public boundary
    EXPECT_THROW(
      (Cached(Matrix<double, Dynamic, Dynamic>(IdentityMatrix<double, Dynamic, Dynamic>(3, 3)))),
      std::invalid_argument);
}

/// @brief checks packed construction and controlled arithmetic through either symmetric cache policy
template <typename Policy, int Order> void check_unified_symmetric_api() {
    using Sym = SymmetricMatrix<double, Order, Policy>;
    Sym matrix(Vector<double, 3> {2., .5, 3.});
    // packed coordinates infer or match order two independently of the cache policy
    EXPECT_EQ(matrix.rows(), 2);
    // packed lower-triangle coordinates reflect into the full symmetric matrix
    EXPECT_DOUBLE_EQ(double(matrix(0, 1)), .5);
    if constexpr (Policy::Flags) {
        const auto& slot = matrix.cache();
        matrix += Dense({1., 0., 0., 1.});
        // compound matrix assignment invalidates the eigenpairs prepared for the old diagonal
        EXPECT_FALSE(slot.valid());
        matrix.cache();
        matrix.cwise() += 2.;
        // coefficientwise arithmetic invalidates the same owner's shared spectral slot
        EXPECT_FALSE(slot.valid());
    } else {
        matrix += Dense({1., 0., 0., 1.});
        matrix.cwise() += 2.;
    }
    const Dense expected({5., 2.5, 2.5, 6.});
    // the two arithmetic paths produce the same independently specified full matrix
    EXPECT_DOUBLE_EQ((matrix - expected).norm(), 0.);
    if constexpr (Order == Dynamic) {
        if constexpr (Policy::Flags) matrix.cache();
        matrix.resize(3);
        // a runtime order change gives the symmetric owner three rows and columns
        EXPECT_EQ(matrix.rows(), 3);
        matrix = Matrix<double, 3, 3>({2., 0., 0., 0., 3., 0., 0., 0., 4.});
        const EVD decomposition(matrix);
        // the resized owner decomposes its new diagonal instead of retaining order-two factors
        EXPECT_NEAR(decomposition.eigenvalues().sum(), 9., 1e-13);
    }
}

// a single public symmetric owner supports the same packed and mutable API with either cache policy
TEST(SymmetricCache, UnifiedPolicyAPI) {
    // fixed uncached owners match explicit packed and full-matrix arithmetic oracles
    check_unified_symmetric_api<Cache::None, 2>();
    // fixed cached owners match the same values and invalidate prepared factors after each compound write
    check_unified_symmetric_api<Cache::Spectral, 2>();
    // dynamic uncached owners additionally match the resized diagonal trace
    check_unified_symmetric_api<Cache::None, Dynamic>();
    // dynamic cached owners rebuild order-three factors instead of retaining their previous order-two cache
    check_unified_symmetric_api<Cache::Spectral, Dynamic>();
    SymmetricMatrix<int, 2> integral(Vector<int, 3> {2, 1, 3});
    integral(0, 1) += 2;
    // the default cache-free policy still supports integral packed coefficients and reflected compound writes
    EXPECT_EQ(int(integral(1, 0)), 3);
}

// mutable and const representations agree on reflected reads while retained writes invalidate their owner
TEST(SymmetricCache, RepresentationReadsAndRetainedWrites) {
    Cached matrix(Vector<double, 3> {2., .5, 3.});
    auto representation = matrix.rep();
    const auto read = std::as_const(matrix).rep();
    // mutable representation reads the mirrored upper entry from the packed lower coefficient
    EXPECT_DOUBLE_EQ(double(representation(0, 1)), .5);
    // const representation uses the same symmetric interpretation of packed storage
    EXPECT_DOUBLE_EQ(read(0, 1), .5);
    const auto& slot = matrix.cache();
    representation(0, 1) = 1.;
    // a representation retained before preparation still invalidates the owner's eigenpairs on write
    EXPECT_FALSE(slot.valid());
    // a retained const representation observes the same shared coefficient after mutation
    EXPECT_DOUBLE_EQ(read(1, 0), 1.);
    const EVD refreshed(matrix);
    // the new symmetric matrix determinant is 2 * 3 - 1 * 1 independently of eigenvalue ordering
    EXPECT_NEAR(refreshed.eigenvalues()[0] * refreshed.eigenvalues()[1], 5., 1e-13);
}

// raw mutable exposure bypasses future cache reuse so retained pointers cannot make eigenpairs stale
TEST(SymmetricCache, RetainedRawOwnerPointer) {
    Cached matrix(Vector<double, 3> {2., 0., 3.});
    const auto& slot = matrix.cache();
    const auto* read = std::as_const(matrix).data();
    // the const packed-storage overload reads the diagonal without invalidating its prepared cache
    EXPECT_DOUBLE_EQ(read[0], 2.);
    // read-only data access preserves the readiness of the already computed eigenpairs
    EXPECT_TRUE(slot.valid());
    auto* writable = matrix.data();
    // requesting a writable alias invalidates previously prepared eigenpairs immediately
    EXPECT_FALSE(slot.valid());
    matrix.cache();
    writable[0] = 7.;
    const auto changed = matrix.eigenvalues();
    // a pointer retained across cache preparation still produces the updated diagonal trace
    EXPECT_NEAR(changed.sum(), 10., 1e-13);
    matrix = Dense({4., 0., 0., 5.});
    matrix.cache();
    writable[0] = 11.;
    const auto replaced = matrix.eigenvalues();
    // same-shape assignment preserves the pointer binding and its permanent cache bypass
    EXPECT_NEAR(replaced.sum(), 16., 1e-13);
    writable[0] = 13.;
    const Cached copy(matrix);
    const auto& copied_slot = copy.cache();
    // copying an exposed owner rebuilds factors from its current coefficients rather than stale source factors
    EXPECT_NEAR(copied_slot.eigenvalues().sum(), 18., 1e-13);
    writable[0] = 17.;
    // independent owning storage keeps its prepared factors valid after a write through the source pointer
    EXPECT_TRUE(copied_slot.valid());
    // the copied coefficients remain independent of later mutations to the original packed storage
    EXPECT_DOUBLE_EQ(copy(0, 0), 13.);
    writable[0] = std::numeric_limits<double>::quiet_NaN();
    // preparation validates raw writes before accepting or reusing eigenpairs for nonfinite coefficients
    EXPECT_THROW(matrix.eigenvalues(), std::invalid_argument);
}

// raw view exposure bypasses one batch slot and copies never inherit stale source eigenpairs
TEST(SymmetricCache, RetainedRawBatchPointer) {
    MatrixBatch<Cached> batch(2);
    batch[0] = Dense({2., 0., 0., 3.});
    batch[1] = Dense({4., 0., 0., 5.});
    auto view = batch[0];
    const auto& slot = view.cache();
    const auto& other_slot = batch[1].cache();
    const auto* read = std::as_const(view).data();
    // const view data access exposes the packed diagonal without changing coefficients
    EXPECT_DOUBLE_EQ(read[0], 2.);
    // read-only access keeps the selected batch slot ready
    EXPECT_TRUE(slot.valid());
    auto* writable = view.data();
    // writable view access invalidates the corresponding batch slot immediately
    EXPECT_FALSE(slot.valid());
    // exposing one coefficient row does not invalidate an unrelated batch element
    EXPECT_TRUE(other_slot.valid());
    view.cache();
    writable[0] = 7.;
    const auto changed = batch[0].eigenvalues();
    // the batch reads the retained pointer's new value after an intervening cache preparation
    EXPECT_NEAR(changed.sum(), 10., 1e-13);
    batch[0] = Dense({4., 0., 0., 5.});
    batch[0].cache();
    writable[0] = 11.;
    const auto copy = batch;
    const EVD copied(copy[0]);
    // same-slot assignment and batch copy cannot reuse factors predating the raw write
    EXPECT_NEAR(copied.eigenvalues().sum(), 16., 1e-13);
    writable[0] = 13.;
    // the batch copy owns independent coefficient rows after a raw mutation of the original
    EXPECT_DOUBLE_EQ(copy[0](0, 0), 11.);
}

// cached SPD logarithms materialize cached symmetric owners whose spectra map to a vector batch
TEST(SymmetricCache, SPDLogarithmBatchEigenvalues) {
    using SPD = SPDMatrix<double, 2, Cache::Log>;
    using Eigenvalues = Vector<double, 2>;
    MatrixBatch<SPD> points(2);
    points[0] = SPD(Vector<double, 3> {2., 0., 3.});
    points[1] = SPD(Vector<double, 3> {4., 0., 5.});
    MatrixBatch<Cached> logs(points.map([](const auto& point) { return matrix_log(point); }));
    MatrixBatch<Eigenvalues> values(logs.map([](const auto& logarithm) { return logarithm.eigenvalues(); }));
    // the first output spectrum has trace log(det(S)) for the independently specified diagonal SPD input
    EXPECT_NEAR(values[0].sum(), std::log(6.), 1e-13);
    // the second batch slot preserves its own logarithmic spectrum and ordering-independent trace
    EXPECT_NEAR(values[1].sum(), std::log(20.), 1e-13);
    logs[0](0, 0) = std::log(7.);
    MatrixBatch<Eigenvalues> changed(logs.map([](const auto& logarithm) { return logarithm.eigenvalues(); }));
    // a new map refreshes the changed log slot before extracting its updated diagonal spectrum
    EXPECT_NEAR(changed[0].sum(), std::log(21.), 1e-13);
    // previously materialized eigenvalue batches retain independent vector values
    EXPECT_NEAR(values[0].sum(), std::log(6.), 1e-13);
}

// direct eigenvectors and exponentials refresh native slots while previously returned owners remain independent
TEST(SymmetricCache, DirectEigenvectorsAndExponentialsRefreshRetainedAliases) {
    Cached matrix(Vector<double, 3> {.2, .1, .4});
    const auto view = std::as_const(matrix).view();
    const auto& slot = matrix.cache();
    matrix(0, 0) = .3;
    // a tracked coefficient change makes the shared factors unready before direct eigenvector extraction
    EXPECT_FALSE(slot.valid());
    const auto vectors = view.eigenvectors();
    const auto values = view.eigenvalues();
    const Dense saved(matrix);
    // const views retain orthogonality in an owning eigenvector type with an unqualified scalar
    static_assert(std::same_as<std::remove_cvref_t<decltype(vectors)>, OrthogonalMatrix<double, 2, 2>>);
    // direct eigenvectors prepare the original owner's shared cache rather than a detached decomposition
    EXPECT_TRUE(slot.valid());
    auto* writable = matrix.data();
    writable[0] = .6;
    const auto exponential = view.exp<Cache::Log>();
    const auto expected = matrix_exp(SymmetricMatrix<double, 2>(matrix));
    // exponential output owners retain the explicitly requested SPD logarithm cache
    static_assert(std::same_as<std::remove_cvref_t<decltype(exponential)>, SPDMatrix<double, 2, Cache::Log>>);
    // exponentiation also refreshes the shared native slot invalidated by writable raw exposure
    EXPECT_TRUE(slot.valid());
    // the cached direct exponential agrees with independently decomposed current source coefficients
    EXPECT_LT((exponential - expected).norm(), 1e-12);
    writable[1] = .2;
    const auto changed_vectors = view.eigenvectors();
    const auto changed_values = view.eigenvalues();
    // a later retained-pointer write refreshes the basis to satisfy the current matrix's eigenvalue equations
    EXPECT_LT((matrix * changed_vectors - changed_vectors * changed_values.as_diagonal()).norm(), 1e-12);
    // earlier owning eigenvectors and values still satisfy the saved source independently of refreshed factors
    EXPECT_LT((saved * vectors - vectors * values.as_diagonal()).norm(), 1e-12);
    writable[2] = .8;
    const auto changed_exp = view.exp();
    const auto expected_changed_exp = matrix_exp(SymmetricMatrix<double, 2>(matrix));
    // a second raw mutation after earlier cache preparation is visible to the next direct exponential
    EXPECT_LT((changed_exp - expected_changed_exp).norm(), 1e-12);
    // the earlier exponential owns its original coefficients after both subsequent raw writes
    EXPECT_LT((exponential - expected).norm(), 1e-12);
    const auto temporary = Cached(Vector<double, 3> {0., 0., 0.}).exp();
    // a temporary symmetric source leaves a checked uncached SPD owner with the analytic unit diagonal
    EXPECT_DOUBLE_EQ(temporary(1, 1), 1.);
}

// direct SPD roots reuse selective prepared quantities while retaining independent policy-selected owners
TEST(SymmetricCache, DirectSPDRootsMatchIndependentUncachedFunctions) {
    using Policy = Cache::Union<Cache::Sqrt, Cache::InverseSqrt>;
    using Point = SPDMatrix<double, 2, Policy>;
    const Point point(Vector<double, 3> {3., .4, 2.});
    const SPDMatrix<double, 2> uncached(point);
    const auto root = point.view().sqrt<Cache::Spectral>();
    const auto inverse_root = point.inv_sqrt<Cache::Log>();
    const auto expected_root = matrix_sqrt(uncached);
    const auto expected_inverse_root = matrix_inv_sqrt(uncached);
    const Dense identity({1, 0, 0, 1});
    const auto vectors = point.eigenvectors();
    const auto values = point.eigenvalues();
    // a root-only SPD cache falls back to decomposition for eigenvectors with correctly paired eigenvalues
    EXPECT_LT((point * vectors - vectors * values.as_diagonal()).norm(), 1e-12);
    // a const SPD view returns a checked owning principal root with the selected spectral policy
    static_assert(std::same_as<std::remove_cvref_t<decltype(root)>, SPDMatrix<double, 2, Cache::Spectral>>);
    // inverse-root output policy is selected independently of the source's prepared root quantities
    static_assert(std::same_as<std::remove_cvref_t<decltype(inverse_root)>, SPDMatrix<double, 2, Cache::Log>>);
    // the direct root from a selective source cache matches an independent uncached eigendecomposition
    EXPECT_LT((root - expected_root).norm(), 1e-12);
    // the direct inverse root matches the same independently supplied coefficient oracle
    EXPECT_LT((inverse_root - expected_inverse_root).norm(), 1e-12);
    // the root and inverse root compose to the identity without assumptions about eigenvector ordering
    EXPECT_LT((root * inverse_root - identity).norm(), 1e-12);
    const auto temporary_root = Point(Vector<double, 3> {4., 0., 9.}).sqrt();
    // a principal root outlives its temporary source and retains the exact square root of nine
    EXPECT_DOUBLE_EQ(temporary_root(1, 1), 3.);
}

// symmetric inverse preserves an indefinite result and refreshes both tracked and raw cache mutations
TEST(SymmetricCache, InversePreservesSymmetryAndRefreshesRawAliases) {
    Cached matrix(Vector<double, 3> {2., 1., -1.});
    const auto result = matrix.inv<Cache::Spectral>();
    const Dense expected({1. / 3., 1. / 3., 1. / 3., -2. / 3.});
    // the inverse of a symmetric matrix retains symmetric ownership and the requested spectral policy
    static_assert(std::same_as<std::remove_cvref_t<decltype(result)>, Cached>);
    // signed reciprocal eigenvalues recover the analytic indefinite inverse
    EXPECT_LT((result - expected).norm(), 1e-12);
    matrix(0, 0) = 3.;
    const auto changed = matrix.inv();
    const Dense snapshot(matrix);
    const auto reference = snapshot.inv();
    // tracked assignment refreshes cached factors before the next inverse
    EXPECT_LT((changed - reference).norm(), 1e-12);
    auto* raw = matrix.data();
    for (double value : {4., 5.}) {
        raw[0] = value;
        const auto actual = matrix.inv();
        const Dense current(matrix);
        const auto independent = current.inv();
        // every retained-pointer mutation is visible despite an intervening cache preparation
        EXPECT_LT((actual - independent).norm(), 1e-12);
    }
    // earlier inverse owners keep their original coefficients after later source mutations
    EXPECT_LT((result - expected).norm(), 1e-12);
    const Cached singular(Vector<double, 3> {0., 0., 1.});
    // a zero cached eigenvalue must reject the undefined inverse
    EXPECT_THROW(singular.inv(), std::domain_error);
    const SymmetricMatrix<double, 2> uncached(singular);
    // the uncached pivoted solve rejects the same singular matrix
    EXPECT_THROW(uncached.inv(), std::domain_error);
    const SymmetricMatrix<double, 1> tiny(Vector<double, 1> {1e-310});
    // an uncached solve also rejects a finite input whose inverse overflows
    EXPECT_THROW(tiny.inv(), std::domain_error);
}

}   // namespace
