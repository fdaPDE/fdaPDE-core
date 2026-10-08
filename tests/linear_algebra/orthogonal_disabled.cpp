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

// disabled debug checks must preserve orthogonalization through input constructors and the matrix method
TEST(OrthogonalDisabled, OrthogonalizesVectorArrayAndMatrixInputs) {
    using namespace fdapde;
    const OrthogonalMatrix<double, Dynamic> from_vector(std::vector<double> {2., 1., 0., 3.}, orthogonalize);
    const double values[] = {2., 1., 0., 3.};
    const OrthogonalMatrix<double, 2> from_array(values, orthogonalize);
    const Matrix<double, 2, 2> directions(values);
    const auto from_matrix = directions.orthogonalize();
    const Matrix<double, 2, 2> identity({1., 0., 0., 1.});
    // the vector input has full rank but is not orthogonal until the constructor executes its algorithm
    EXPECT_NEAR((from_vector.transpose() * from_vector - identity).norm(), 0., 1e-12);
    // the fixed array overload must execute the same algorithm independently of assertion mode
    EXPECT_NEAR((from_array.transpose() * from_array - identity).norm(), 0., 1e-12);
    // transpose-based inversion is valid only after the orthogonal invariant has been established
    EXPECT_NEAR((from_array.inv() * from_array - identity).norm(), 0., 1e-12);
    // the matrix method executes orthogonalization with internal debug checks disabled
    EXPECT_NEAR((from_matrix.transpose() * from_matrix - identity).norm(), 0., 1e-12);
}

// invariant validation remains active for public input even when internal debug checks are disabled
TEST(OrthogonalDisabled, RejectsInvalidInputs) {
    using namespace fdapde;
    using Owner = OrthogonalMatrix<double, 2>;
    const Matrix<double, 2, 2> dependent({1., 2., 2., 4.});
    // the matrix method rejects collinear columns instead of publishing an invalid orthogonal owner
    EXPECT_THROW(static_cast<void>(dependent.orthogonalize()), std::invalid_argument);
    const double coefficients[] = {1., 2., 2., 4.};
    // array orthogonalization also rejects a basis whose columns are linearly dependent
    EXPECT_THROW(static_cast<void>(Owner(coefficients, orthogonalize)), std::invalid_argument);
    // vector orthogonalization cannot bypass the same rank requirement
    EXPECT_THROW(
      static_cast<void>(OrthogonalMatrix<double, Dynamic>(std::vector<double> {1., 2., 2., 4.}, orthogonalize)),
      std::invalid_argument);
    // direct construction requires already orthonormal coefficients
    EXPECT_THROW(static_cast<void>(Owner(dependent)), std::invalid_argument);
    // direct array construction enforces the same orthonormality condition
    EXPECT_THROW(static_cast<void>(Owner(coefficients)), std::invalid_argument);
    // incomplete packed input is rejected before any coefficient read
    EXPECT_THROW(static_cast<void>(Owner(std::vector<double> {1., 0., 1.})), std::invalid_argument);
    const Matrix<double, Dynamic, Dynamic> rectangular(2, 3);
    // runtime rectangular dimensions are invalid even without debug shape checks
    EXPECT_THROW(static_cast<void>(rectangular.orthogonalize()), std::invalid_argument);
    const Matrix<double, Dynamic, Dynamic> larger(IdentityMatrix<double, Dynamic, Dynamic>(3, 3));
    // a fixed owner must not silently crop a larger dynamically shaped source
    EXPECT_THROW(static_cast<void>(Owner(larger)), std::invalid_argument);
    const Matrix<double, 2, 2> nonfinite({std::numeric_limits<double>::quiet_NaN(), 0., 0., 1.});
    // nonfinite coefficients cannot establish orthogonality
    EXPECT_THROW(static_cast<void>(Owner(nonfinite)), std::invalid_argument);
    // external storage is validated before it is exposed through an orthogonal view
    EXPECT_THROW((static_cast<void>(OrthogonalMatrixView<const double, 2>(dependent.data()))), std::invalid_argument);
    // external storage cannot be null for a nonempty orthogonal view
    EXPECT_THROW(
      (static_cast<void>(OrthogonalMatrixView<const double, 2>(static_cast<const double*>(nullptr)))),
      std::invalid_argument);
    const auto forged = internals::orthogonal_cast(dependent);
    // merely carrying the expression tag does not certify external coefficients
    EXPECT_THROW(static_cast<void>(Owner(forged)), std::invalid_argument);
}

// failed replacements preserve both owned coefficients and view bindings without debug assertions
TEST(OrthogonalDisabled, PreservesRejectedReplacements) {
    using namespace fdapde;
    const Matrix<double, 2, 2> valid({0., -1., 1., 0.});
    OrthogonalMatrix<double, 2> owner(valid);
    auto view = owner.view();
    const Matrix<double, 2, 2> invalid({1., 1., 1., 1.});
    // invalid owned replacement fails before committing coefficients
    EXPECT_THROW(owner = invalid, std::invalid_argument);
    // the rejected owner assignment retains the known quarter-turn matrix
    EXPECT_EQ(owner.data(), valid);
    // invalid borrowed replacement is also validated before touching the owner's buffer
    EXPECT_THROW(view = invalid, std::invalid_argument);
    // the rejected view assignment leaves all owner coefficients unchanged
    EXPECT_EQ(owner.data(), valid);
    const Matrix<double, Dynamic, Dynamic> larger {IdentityMatrix<double, 3, 3>()};
    // even an orthogonal replacement must fit the view's established shape
    EXPECT_THROW(view = larger, std::invalid_argument);
    // failed shape validation does not modify the borrowed coefficients
    EXPECT_EQ(owner.data(), valid);
}

// unavailable QR orthogonal storage stays inaccessible even when internal debug checks are disabled
TEST(OrthogonalDisabled, QRRejectsUnavailableFactors) {
    using namespace fdapde;
    HouseholderQR<double, 2, 2> qr;
    const Vector<double, 2> rhs {1., 2.};
    // solve cannot access unavailable factors when debug-only assertions are erased
    EXPECT_THROW(qr.solve(rhs), std::logic_error);
    // default construction cannot expose an orthogonal owner that has not been built
    EXPECT_THROW(static_cast<void>(qr.Q()), std::logic_error);
    const Matrix<double, 2, 2> zero = Matrix<double, 2, 2>::Zero();
    qr.compute(zero);
    // numerical rank remains a permanent solve precondition in release builds
    EXPECT_THROW(qr.solve(rhs), std::domain_error);
    const auto& q = qr.Q();
    // rank-zero input still produces a full orthogonal basis whose inverse composes to identity
    EXPECT_EQ((Matrix<double, 2, 2>(q.inv() * q)), (IdentityMatrix<double, 2, 2>()));
    const Matrix<double, 2, 2> nonfinite({std::numeric_limits<double>::quiet_NaN(), 0., 0., 1.});
    // nonfinite input prevents a new factor from being published
    EXPECT_THROW(qr.compute(nonfinite), std::invalid_argument);
    // a failed recomputation cannot expose the previously computed factor
    EXPECT_THROW(static_cast<void>(qr.Q()), std::logic_error);
}

// unavailable EVD factors stay inaccessible even when internal debug checks are disabled
TEST(OrthogonalDisabled, EVDRejectsUnavailableFactors) {
    using namespace fdapde;
    EVD<SymmetricMatrix<double, 2>> evd;
    const Vector<double, 2> rhs {1., 0.};
    // solve cannot consume unavailable eigenpairs when debug-only checks are erased
    EXPECT_THROW(evd.solve(rhs), std::logic_error);
    // default construction cannot expose an orthogonal eigenvector owner that has not been built
    EXPECT_THROW(static_cast<void>(evd.eigenvectors()), std::logic_error);
    // default construction cannot expose an uncomputed spectrum
    EXPECT_THROW(static_cast<void>(evd.eigenvalues()), std::logic_error);
    SymmetricMatrix<double, 2> zero;
    evd.compute(zero);
    // zero eigenvalues remain an always-active solve failure even for a completed decomposition
    EXPECT_THROW(evd.solve(rhs), std::domain_error);
    const auto& basis = evd.eigenvectors();
    // repeated zero eigenvalues still produce a full orthogonal basis whose inverse composes to identity
    EXPECT_EQ((Matrix<double, 2, 2>(basis.inv() * basis)), (IdentityMatrix<double, 2, 2>()));
    zero(0, 0) = std::numeric_limits<double>::quiet_NaN();
    // nonfinite coefficients prevent new eigenpairs from being published
    EXPECT_THROW(evd.compute(zero), std::invalid_argument);
    // a failed recomputation cannot expose the previous eigenvector factor
    EXPECT_THROW(static_cast<void>(evd.eigenvectors()), std::logic_error);
    // a failed recomputation cannot expose previous eigenvalues
    EXPECT_THROW(static_cast<void>(evd.eigenvalues()), std::logic_error);
}

// generic symmetric algorithms retain public validation when debug-only preconditions are erased
TEST(OrthogonalDisabled, DenseSymmetricAlgorithmsRejectInvalidInputs) {
    using namespace fdapde;
    const Matrix<double, 2, 2> asymmetric({2., 3., 1., 2.});
    // EVD validates both triangles of a generic matrix in release builds too
    EXPECT_THROW((EVD(asymmetric)), std::invalid_argument);
    // Cholesky retains the same always-active symmetry boundary
    EXPECT_THROW((Cholesky(asymmetric)), std::invalid_argument);
    const Matrix<double, 2, 2> singular({1., 1., 1., 1.});
    // nonpositive pivots are algorithm failures rather than debug-only preconditions
    EXPECT_THROW((Cholesky(singular)), std::domain_error);
    Cholesky<Matrix<double, 2, 2>> empty;
    // default construction cannot expose uninitialized factor storage even without debug checks
    EXPECT_THROW(empty.L(), std::logic_error);
}
