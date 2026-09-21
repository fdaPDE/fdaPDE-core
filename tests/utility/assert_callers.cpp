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

#include <fdaPDE/core.h>
#include <gtest/gtest.h>

#include <typeinfo>

namespace {
/// @brief checks the exact exception type and diagnostic produced by a failed contract
template <typename Exception, typename Callable> void expect_failure(Callable&& call, const char* message) {
    try {
        call();
        // a rejected operation must throw before returning to the caller
        FAIL() << "operation did not throw";
    } catch (const Exception& error) {
        // checks the concrete type so a generic logic_error cannot hide a classification regression
        EXPECT_EQ(typeid(error), typeid(Exception));
        // checks which independent precondition rejected the operation
        EXPECT_STREQ(error.what(), message);
    }
}
}   // namespace

// verifies row and column bounds produce distinct diagnostics through mutable and const views
TEST(AssertionCallers, ArrayIndicesIdentifyTheFailingAxis) {
    fdapde::MdArray<int, fdapde::MdExtents<fdapde::Dynamic, fdapde::Dynamic>> data(2, 3);
    const auto& const_data = data;
    // checks the lower row bound before creating a mutable view
    expect_failure<std::out_of_range>([&] { data.row(-1); }, "row index must be nonnegative");
    // checks the upper row bound before creating a mutable view
    expect_failure<std::out_of_range>([&] { data.row(2); }, "row index out of range");
    // checks the lower column bound before creating a mutable view
    expect_failure<std::out_of_range>([&] { data.col(-1); }, "column index must be nonnegative");
    // checks the upper column bound before creating a mutable view
    expect_failure<std::out_of_range>([&] { data.col(3); }, "column index out of range");
    // verifies the const overload preserves the row diagnostic
    expect_failure<std::out_of_range>([&] { const_data.row(2); }, "row index out of range");
    // verifies the const overload preserves the column diagnostic
    expect_failure<std::out_of_range>([&] { const_data.col(3); }, "column index out of range");
    // confirms valid boundary indices still allow access
    EXPECT_NO_THROW(data(1, 2) = 7);
}

// verifies binary operations report row and column mismatches separately
TEST(AssertionCallers, BinaryOperandsIdentifyTheMismatchedDimension) {
    fdapde::BinaryMatrix<fdapde::Dynamic, fdapde::Dynamic> lhs(2, 3), wrong_rows(3, 3), wrong_cols(2, 4);
    // rejects a row mismatch without requiring a column mismatch
    expect_failure<std::invalid_argument>([&] { (void)(lhs | wrong_rows); }, "operand row counts must match");
    // reaches the column check after the row check succeeds
    expect_failure<std::invalid_argument>([&] { (void)(lhs | wrong_cols); }, "operand column counts must match");
    // verifies binary matrix access is classified as an index failure
    expect_failure<std::out_of_range>([&] { lhs.set(2, 0); }, "row index out of range");
    // verifies the column access check has its own diagnostic
    expect_failure<std::out_of_range>([&] { lhs.set(0, 3); }, "column index out of range");
}

// verifies spline indices and construction arguments use different exception categories
TEST(AssertionCallers, SplineChecksSeparateIndexOrderAndKnotCount) {
    std::vector<double> knots {0, 1, 2};
    // rejects an invalid index before evaluating the remaining constructor checks
    expect_failure<std::out_of_range>([&] { fdapde::Spline spline(knots, -1, 1); }, "spline index must be nonnegative");
    // identifies a negative polynomial order as an invalid argument
    expect_failure<std::invalid_argument>(
      [&] { fdapde::Spline spline(knots, 0, -1); }, "spline order must be nonnegative");
    // identifies a knot vector too short for the requested order
    expect_failure<std::invalid_argument>(
      [&] { fdapde::Spline spline(knots, 0, 3); }, "knot count must exceed the spline order");
    // reaches the upper index check after the other arguments pass
    expect_failure<std::out_of_range>(
      [&] { fdapde::Spline spline(knots, 3, 1); }, "spline index exceeds the knot vector");
    // confirms valid construction remains accepted
    EXPECT_NO_THROW(fdapde::Spline(knots, 0, 1));
}

// verifies mesh construction reports each malformed input independently before geometric assembly
TEST(AssertionCallers, TriangulationChecksSeparateInputShapes) {
    Eigen::MatrixXd nodes(3, 2);
    nodes << 0, 0, 1, 0, 0, 1;
    Eigen::MatrixXi cells(1, 3), boundary = Eigen::MatrixXi::Ones(3, 1);
    cells << 0, 1, 2;
    auto construct = [](const Eigen::MatrixXd& n, const Eigen::MatrixXi& c, const Eigen::MatrixXi& b) {
        fdapde::Triangulation<2, 2> mesh(n, c, b);
    };
    // the empty-node check must run before checks that use node coordinates
    expect_failure<std::invalid_argument>(
      [&] { construct(Eigen::MatrixXd(0, 2), cells, boundary); }, "triangulation nodes must not be empty");
    // reaches the coordinate dimension check with a nonempty node matrix
    expect_failure<std::invalid_argument>(
      [&] { construct(Eigen::MatrixXd(3, 1), cells, boundary); },
      "node coordinate dimension must match the embedding dimension");
    // an empty cell matrix must fail before minCoeff is evaluated
    expect_failure<std::invalid_argument>(
      [&] { construct(nodes, Eigen::MatrixXi(0, 3), boundary); }, "triangulation cells must not be empty");
    // reaches the cell width check with a nonempty cell matrix
    expect_failure<std::invalid_argument>(
      [&] { construct(nodes, Eigen::MatrixXi(1, 2), boundary); }, "cell width must match the number of nodes per cell");
    // identifies an incorrect number of boundary markers
    expect_failure<std::invalid_argument>(
      [&] { construct(nodes, cells, Eigen::MatrixXi(2, 1)); }, "boundary marker count must match the number of nodes");
    // identifies an incorrect boundary marker orientation
    expect_failure<std::invalid_argument>(
      [&] { construct(nodes, cells, Eigen::MatrixXi(3, 2)); }, "boundary markers must form a column vector");
    // confirms the valid inputs still produce a mesh
    EXPECT_NO_THROW(construct(nodes, cells, boundary));
}

// verifies invalid marker values are distinct from missing marker initialization
TEST(AssertionCallers, CellFilteringSeparatesArgumentsFromObjectState) {
    fdapde::Triangulation<1, 1> interval(0.0, 1.0, 3);
    // a negative marker other than the all-cells sentinel is an invalid argument
    expect_failure<std::invalid_argument>(
      [&] { interval.cells_begin(-3); }, "cell marker must be nonnegative or TriangulationAll");
    // a valid marker cannot be used before marker storage is initialized
    expect_failure<std::logic_error>(
      [&] { interval.cells_begin(1); }, "cell markers must be initialized before filtering");
    // the sentinel bypasses marker initialization for both iterator endpoints
    EXPECT_NO_THROW({
        interval.cells_begin();
        interval.cells_end();
    });
}

// verifies argument exceptions propagate through constructors instead of terminating in noexcept
TEST(AssertionCallers, CheckedConstructorsPropagateExceptions) {
    Eigen::SparseMatrix<double> rectangular(2, 3), empty(0, 0);
    // checks that the factorization constructor propagates its shape diagnostic
    expect_failure<std::invalid_argument>(
      [&] { fdapde::FSPAI<Eigen::SparseMatrix<double>> factor(rectangular); }, "FSPAI requires a square matrix");
    // reaches the nonempty check after the square-matrix check succeeds
    expect_failure<std::invalid_argument>(
      [&] { fdapde::FSPAI<Eigen::SparseMatrix<double>> factor(empty); }, "FSPAI requires a nonempty matrix");
    // verifies polygon construction propagates an invalid argument to its caller
    expect_failure<std::invalid_argument>(
      [] { fdapde::Polygon<2, 2> polygon(Eigen::MatrixXd(0, 2)); }, "polygon nodes must not be empty");
    fdapde::Triangulation<1, 1> interval;
    // verifies geometric assembly helpers also propagate constructor checks
    expect_failure<std::invalid_argument>(
      [&] { fdapde::CellDiameter diameter(interval); }, "triangulation must contain nodes");
}

// verifies constexpr-capable callers retain compile-time use and classify runtime failures
TEST(AssertionCallers, ConstantEvaluationAndRuntimeExceptionTypesAgree) {
    // evaluates the successful mathematical operation at compile time
    static_assert(fdapde::factorial(5) == 120);
    // distinguishes an undefined mathematical input from an index or shape error
    expect_failure<std::domain_error>(
      [] { (void)fdapde::factorial(-1); }, "factorial is undefined for negative integers");
    // checks the typed replacement of a constexpr matrix constructor precondition
    expect_failure<std::invalid_argument>(
      [] { fdapde::Matrix<double, 2, 2> matrix(std::vector<double>(3)); },
      "coefficient count must match the matrix size");
}

// verifies name lookup failures are classified like keyed container access
TEST(AssertionCallers, MissingColumnsThrowOutOfRange) {
    fdapde::internals::scalar_data_layer data;
    // a missing named column must fail before constructing a column view
    expect_failure<std::out_of_range>([&] { data.col<double>("missing"); }, "column name not found");
    // a missing column used by the null mask has the same lookup category
    expect_failure<std::out_of_range>([&] { data.nan(std::string("missing")); }, "column name not found");
}

// verifies the dot product checks both output dimensions instead of comparing a row count with itself
TEST(AssertionCallers, DotProductChecksBothOperandDimensions) {
    using Field = fdapde::MatrixField<2, fdapde::Dynamic, fdapde::Dynamic>;
    Field lhs(2, 2, 3), wrong_rows(2, 3, 3), wrong_cols(2, 2, 4);
    // rejects unequal row counts even when column counts match
    expect_failure<std::invalid_argument>(
      [&] { (void)fdapde::dot(lhs, wrong_rows); }, "dot product row counts must match");
    // reports the column mismatch independently
    expect_failure<std::invalid_argument>(
      [&] { (void)fdapde::dot(lhs, wrong_cols); }, "dot product column counts must match");
    // confirms equal operand shapes still pass construction
    EXPECT_NO_THROW((void)fdapde::dot(lhs, lhs));
}
