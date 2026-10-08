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

#include <fdaPDE/geometric_finite_elements.h>
#include <fdaPDE/geometric_splines.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using Policy = Cache::Union<Cache::Spectral, Cache::Log, Cache::LogDividedDifferences>;
using SPD = SPDMatrix<double, 2, Policy>;
using Sym = SymmetricMatrix<double, 2>;
using LE = manifold::LogEuclideanGeometry<SPD>;
using Locations = MatrixBatch<Vector<double, 1>>;
/// @brief supplies nonuniform spans on a nonunit physical interval
Triangulation<1, 1> interval() {
    Eigen::VectorXd knots(5);
    knots << -1., -.6, .2, 1.3, 2.;
    return Triangulation<1, 1>(knots);
}
/// @brief defines varying noncommuting logarithmic control coefficients
MatrixBatch<SPD> controls(int count) {
    MatrixBatch<Sym> logarithms(count);
    for (int i = 0; i < count; ++i) logarithms[i] = Sym(Vector<double, 3> {.1 + .08 * i, .03 - .01 * i, -.2 + .04 * i});
    return logarithms.exp<Policy>();
}
/// @brief compares full Frobenius coefficients without retaining temporary expressions
double error(const auto& a, const auto& b) { return Matrix<double, 2, 2>(a - b).norm(); }
/// @brief obtains an independent scalar spline for each packed logarithmic coefficient
Sym scalar_oracle(const auto& scalar, const auto& points, double x) {
    Vector<double, 3> packed;
    const Eigen::Matrix<double, 1, 1> location = Eigen::Matrix<double, 1, 1>::Constant(x);
    for (int coefficient = 0; coefficient < 3; ++coefficient) {
        double value = 0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            const Sym logarithm(points[i].log());
            const double entry = coefficient == 0 ? logarithm(0, 0) :
                                 coefficient == 1 ? logarithm(1, 0) :
                                                    logarithm(1, 1);
            value += scalar.physical_basis()[i](location[0]) * entry;
        }
        packed[coefficient] = value;
    }
    return Sym(packed);
}
// spline means reproduce native scalar B-spline fields in logarithmic coordinates at spans, knots and endpoints
TEST(GeometricSpace, SplineScalarOracleAndBoundarySupport) {
    const auto domain = interval();
    for (int degree : {0, 1, 3, 5}) {
        const BsSpace scalar(domain, degree);
        const GeometricSpace space(scalar, LE {});
        const auto points = controls(space.n_dofs());
        const GeometricFunction field(space, points);
        // wrapping a scalar space preserves its exact binding and numbering rather than enumerating a copy
        EXPECT_EQ(&space.scalar_space(), &scalar);
        for (double x : {-1., -.6, -.17, .2, .7, 1.3, 2.}) {
            const Eigen::Matrix<double, 1, 1> location = Eigen::Matrix<double, 1, 1>::Constant(x);
            const SPD actual(field(location));
            const SPD expected(matrix_exp(scalar_oracle(scalar, points, x)));
            // the geometric spline equals three independently evaluated scalar logarithmic splines
            EXPECT_LT(error(actual, expected), 3e-12);
        }
        // clamped endpoint basis weights select the first control exactly, including degree zero
        EXPECT_LT(error(field.result(Eigen::Matrix<double, 1, 1>::Constant(-1.)).value, points[0]), 1e-13);
        // clamped endpoint basis weights select the last control exactly, including degree zero
        EXPECT_LT(
          error(field.result(Eigen::Matrix<double, 1, 1>::Constant(2.)).value, points[points.size() - 1]), 1e-13);
    }
}
// one spatial preparation supports sequential, parallel and updated spline coefficients with owning output batches
TEST(GeometricSpace, SplinePreparedEvaluationAndReplacement) {
    parallel_set_num_threads(3);
    const auto domain = interval();
    const BsSpace scalar(domain, 3);
    const GeometricSpace space(scalar, LE {}), other_space(scalar, LE {});
    GeometricFunction field(space, controls(space.n_dofs()));
    Locations locations(97);
    for (std::size_t i = 0; i < locations.size(); ++i)
        locations[i] = Vector<double, 1> {-1. + 3. * i / (locations.size() - 1)};
    const auto Psi = space.prepare_evaluation(locations, execution_par);
    const auto first = Psi(field);
    const auto parallel = field.eval_at(locations, execution_par);
    for (std::size_t i = 0; i < locations.size(); ++i) {
        // both execution policies evaluate the same active controls in the supplied location order
        EXPECT_LT(error(first[i], parallel[i]), 1e-13);
    }
    const auto x = Eigen::Matrix<double, 1, 1>::Constant(.4);
    const SPD original(field(x));
    const auto borrowed = field(x);
    (void)field(Eigen::Matrix<double, 1, 1>::Constant(1.7));
    // preparing another span cannot invalidate an expression that borrows a previous span
    EXPECT_LT(error(SPD(borrowed), original), 1e-13);
    auto logarithms = field.coeff().log();
    const Sym shift(Vector<double, 3> {std::log(2.), 0., std::log(2.)});
    logarithms.assign(logarithms + shift, execution_par);
    field.set_coeff(logarithms.exp<Policy>(execution_par));
    // replacing controls discards coefficient-dependent local span data
    EXPECT_EQ(field.prepared_cells(), 0);
    const auto updated = Psi(field, execution_par);
    for (std::size_t i = 0; i < first.size(); ++i) {
        // the retained spatial plan uses current controls while previous owning values remain unchanged
        EXPECT_LT(error(updated[i], 2. * first[i]), 4e-12);
    }
    const GeometricFunction other(other_space, controls(other_space.n_dofs()));
    // a preparation rejects a different space instance even when both borrow the same scalar basis
    EXPECT_THROW(Psi(other), std::invalid_argument);
    const auto empty = space.prepare_evaluation(Locations {}, execution_par)(field, execution_par);
    // empty evaluation preserves the declared output matrix order without constructing fake controls
    EXPECT_TRUE(empty.empty() && empty.rows() == 2 && empty.cols() == 2);
}
// spline coefficient and weight derivatives agree with physical finite differences and their adjoint identity
TEST(GeometricSpace, SplineDifferentialsAndControlOrdering) {
    const auto domain = interval();
    const BsSpace scalar(domain, 3);
    const GeometricSpace space(scalar, LE {});
    auto points = controls(space.n_dofs());
    const GeometricFunction field(space, points);
    const double x = .47, step = 1e-6;
    const Eigen::Matrix<double, 1, 1> location = Eigen::Matrix<double, 1, 1>::Constant(x);
    const int span = scalar.triangulation().locate(location);
    const auto dofs = scalar.dof_handler().active_dofs(span);
    std::vector<double> weight_direction;
    for (int id : dofs) weight_direction.push_back(scalar.physical_basis()[id].gradient(1)(x));
    const auto linearization = field.linearization(location);
    const auto spatial = linearization.weight_jvp(weight_direction);
    const SPD plus(field(Eigen::Matrix<double, 1, 1>::Constant(x + step)));
    const SPD minus(field(Eigen::Matrix<double, 1, 1>::Constant(x - step)));
    const Sym numerical((plus - minus) / (2 * step));
    // physical B-spline derivatives compose with the weighted geometric differential
    EXPECT_LT(error(spatial, numerical), 2e-8);
    std::vector<Sym> directions;
    for (int id : dofs) directions.emplace_back(points[id]);
    const auto nodal = linearization.nodal_jvp(std::span<const Sym>(directions));
    const SPD value(field(location));
    // common scaling of all active controls gives the derivative of exp(t) times the interpolated tensor
    EXPECT_LT(error(nodal, value), 2e-12);
    const Sym covector(Vector<double, 3> {.2, -.05, .1});
    const auto adjoint = linearization.nodal_vjp(covector);
    double paired = 0;
    for (std::size_t i = 0; i < dofs.size(); ++i)
        paired += space.geometry().inner_product(points[dofs[i]], directions[i], adjoint[i]);
    // local control derivatives and their metric adjoint use exactly the same active-index ordering
    EXPECT_NEAR(paired, space.geometry().inner_product(value, nodal, covector), 3e-12);
}
// commuting diagonal controls give an analytic affine-invariant oracle without hiding the iterative result status
TEST(GeometricSpace, SplineAffineInvariantMeanAndFailure) {
    const auto domain = interval();
    const BsSpace scalar(domain, 3);
    using Geometry = manifold::AffineInvariantGeometry<SPD>;
    const GeometricSpace space(scalar, Geometry {});
    MatrixBatch<SPD> points(space.n_dofs());
    for (std::size_t i = 0; i < points.size(); ++i)
        points[i] = SPD(Vector<double, 3> {std::exp(.1 * i), 0., std::exp(-.04 * i)});
    const GeometricFunction field(space, points);
    const auto x = Eigen::Matrix<double, 1, 1>::Constant(.47);
    const auto result = field.result(x);
    // commuting SPD inputs admit a unique mean that the intrinsic solver must certify
    ASSERT_TRUE(result.converged());
    // the commuting affine-invariant mean equals the componentwise geometric mean defined by the scalar basis
    EXPECT_LT(error(result.value, matrix_exp(scalar_oracle(scalar, points, x[0]))), 2e-9);
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.max_iterations = 1;
    options.mean.solver.gradient_tolerance = 0;
    const GeometricFunction failing(space, controls(space.n_dofs()), options);
    Locations locations(1);
    locations[0] = Vector<double, 1> {.47};
    const auto Psi = space.prepare_evaluation(locations);
    // a failed intrinsic mean cannot be silently materialized through parallel prepared evaluation
    EXPECT_THROW(Psi(failing, execution_par), std::runtime_error);
}
// rotation-valued spline controls preserve their native type and match a small-angle scalar spline oracle
TEST(GeometricSpace, SplineRotationMean) {
    const auto domain = interval();
    const BsSpace scalar(domain, 3);
    using Rotation = RotationMatrix<double, 2>;
    const manifold::SOGeometry<double, 2> geometry;
    const GeometricSpace space(scalar, geometry);
    MatrixBatch<Rotation> points(space.n_dofs());
    Eigen::VectorXd angles(space.n_dofs());
    for (int i = 0; i < space.n_dofs(); ++i) {
        angles[i] = -.1 + .04 * i;
        points[i] = Rotation(
          Matrix<double, 2, 2>({std::cos(angles[i]), -std::sin(angles[i]), std::sin(angles[i]), std::cos(angles[i])}));
    }
    const GeometricFunction field(space, points);
    const auto x = Eigen::Matrix<double, 1, 1>::Constant(.47);
    double expected = 0;
    for (int i = 0; i < angles.size(); ++i) expected += scalar.physical_basis()[i](x[0]) * angles[i];
    const auto result = field.result(x);
    // all controls lie in one small-angle branch and the intrinsic mean must converge
    ASSERT_TRUE(result.converged());
    const Rotation reference(
      Matrix<double, 2, 2>({std::cos(expected), -std::sin(expected), std::sin(expected), std::cos(expected)}));
    // the resolved SO(2) mean equals the scalar B-spline angle within the selected branch
    EXPECT_LT(error(Rotation(field(x)), reference), 2e-9);
    Locations locations(1);
    locations[0] = Vector<double, 1> {x[0]};
    const auto evaluated = field.eval_at(locations, execution_par);
    // batch materialization retains verified rotation owners rather than SPD or dense destinations
    static_assert(std::same_as<typename decltype(evaluated)::MatrixType, Rotation>);
}
// Cheeger scalar controls are interpolated with the same spline weights and updates preserve spatial preparation
TEST(GeometricSpace, SplineCheegerMetricControls) {
    const auto domain = interval();
    const BsSpace scalar(domain, 3);
    using Geometry = manifold::CheegerLogEuclideanGeometry<SPD>;
    std::vector<double> rho(scalar.n_dofs());
    for (std::size_t i = 0; i < rho.size(); ++i) rho[i] = .2 + .01 * i;
    const GeometricSpace space(scalar, Geometry(.25), std::span<const double>(rho));
    GeometricFunction field(space, controls(space.n_dofs()));
    const auto x = Eigen::Matrix<double, 1, 1>::Constant(.47);
    std::vector<double> weights(space.n_dofs());
    double local_rho = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = scalar.physical_basis()[i](x[0]);
        local_rho += weights[i] * rho[i];
    }
    const auto expected = gfe::p1_geodesic_value(space.geometry().with_rho(local_rho), field.coeff(), weights);
    const auto actual = field.result(x);
    // the localized spline mean and the full-basis metric oracle must both be certified
    ASSERT_TRUE(expected.converged() && actual.converged());
    // rho and matrix coefficients share exactly the same physical B-spline basis weights
    EXPECT_LT(error(actual.value, expected.value), 2e-8);
    Locations locations(1);
    locations[0] = Vector<double, 1> {x[0]};
    const auto Psi = space.prepare_evaluation(locations);
    field.set_rho({});
    const auto constant = gfe::p1_geodesic_value(space.geometry(), field.coeff(), weights);
    // removing the scalar override restores the geometry's constant rho in an existing spatial plan
    ASSERT_TRUE(constant.converged());
    const auto restored = Psi(field, execution_par);
    // metric replacement invalidates local span data without invalidating retained spatial basis data
    EXPECT_LT(error(restored[0], constant.value), 2e-8);
}

// lifetime guards and invalid input checks apply to both scalar discretizations through the shared facade
TEST(GeometricSpace, SplineLifetimeAndInputContracts) {
    using Scalar = BsSpace<Triangulation<1, 1>>;
    using Space = GeometricSpace<const Scalar, LE>;
    // explicit const template arguments cannot allow borrowing a temporary scalar spline space
    static_assert(!std::is_constructible_v<Space, Scalar&&, LE>);
    using Function = GeometricFunction<const Space, SPD>;
    // fields cannot borrow an expiring geometric space even with explicit const template arguments
    static_assert(!std::is_constructible_v<Function, Space&&, MatrixBatch<SPD>>);
    const auto domain = interval();
    const Scalar scalar(domain, 3);
    const Space space(scalar, LE {});
    const GeometricFunction field(space, controls(space.n_dofs()));
    // outside coordinates are rejected before selecting control coefficients
    EXPECT_THROW(field.result(Eigen::Matrix<double, 1, 1>::Constant(2.1)), std::invalid_argument);
    // nonfinite coordinates are rejected before searching for a knot span
    EXPECT_THROW(
      field.result(Eigen::Matrix<double, 1, 1>::Constant(std::numeric_limits<double>::quiet_NaN())),
      std::invalid_argument);
    Locations locations(1);
    locations[0] = Vector<double, 1> {-1.1};
    // spatial preparation applies the same domain checks under the parallel policy
    EXPECT_THROW(space.prepare_evaluation(locations, execution_par), std::invalid_argument);
    // owning coefficients must cover exactly the scalar spline control DOFs
    EXPECT_THROW(GeometricFunction(space, MatrixBatch<SPD>(space.n_dofs() - 1)), std::invalid_argument);
}
}   // namespace
