// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
/// @brief checks FEM stencil assembly against independent gradients on the unit right triangle
TEST(GeometricSmoothingFEM, NativeSpaceAndScalarAssembly) {
    Eigen::Matrix<double, 3, 2> vertices;
    vertices << 0, 0, 1, 0, 0, 1;
    Eigen::Matrix<int, 1, 3> cells;
    cells << 0, 1, 2;
    const Triangulation<2, 2> mesh(vertices, cells, Eigen::Matrix<int, 3, 1>::Ones());
    using LE = manifold::LogEuclideanSPDGeometry<double, 2>;
    const fdapde::FeSpace scalar_source_space(mesh, fdapde::P1<1>);
    const GeometricSpace space(scalar_source_space, LE {});
    const auto packet = gfe::p1_fem_cell_quadrature(space, 0, QS2DP4);
    const auto stencil = gfe::p1_lumped_laplacian_stencil(space, QS2DP4);
    // P1 lumping divides area 1/2 equally between the three scalar nodal DOFs
    for (double mass : stencil.lumped_masses) EXPECT_NEAR(mass, 1. / 6, 1e-15);
    // orthogonal gradients at the last two vertices give zero stiffness between them
    ASSERT_EQ(stencil.edges.size(), 2u);
    // the first basis gradient has inner product -1 with each remaining gradient and area is 1/2
    EXPECT_NEAR(stencil.edges[0].stiffness, -.5, 1e-15);
    // quadrature integration weights reproduce the exact area
    EXPECT_NEAR(std::accumulate(packet.integration_weights.begin(), packet.integration_weights.end(), 0.), .5, 1e-15);
    const FeSpace scalar(mesh, P1<1>);
    const auto reference = gfe::p1_lumped_laplacian_stencil(scalar, QS2DP4);
    // wrapping a scalar space must not change masses or DOF order
    EXPECT_EQ(reference.lumped_masses, stencil.lumped_masses);
    // out-of-mesh cells are rejected before accessing the DOF handler
    EXPECT_THROW(gfe::p1_fem_cell_quadrature(space, 1, QS2DP4), std::out_of_range);
}
/// @brief checks the recovered lumped tension against the smooth Neumann eigenfunction energy
TEST(GeometricSmoothingFEM, NeumannRefinement) {
    constexpr double pi = 3.14159265358979323846;
    double previous = std::numeric_limits<double>::infinity();
    for (int side : {9, 17, 33}) {
        const auto mesh = Triangulation<2, 2>::UnitSquare(side);
        using LE = manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
        const fdapde::FeSpace scalar_source_space(mesh, fdapde::P1<1>);
        const GeometricSpace space(scalar_source_space, LE {});
        const auto stencil = gfe::p1_lumped_laplacian_stencil(space, QS2DP4);
        MatrixBatch<LE::Point> coefficients(space.n_dofs());
        for (int i = 0; i < space.n_dofs(); ++i) {
            const double h = std::cos(pi * mesh.nodes()(i, 0)) * std::cos(pi * mesh.nodes()(i, 1));
            coefficients[i] = Matrix<double, 2, 2>({std::exp(h), 0., 0., 1.});
        }
        const auto energy = gfe::p1_discrete_tension_value(LE {}, coefficients, stencil);
        const double relative = std::abs(energy.value / (.5 * pi * pi * pi * pi) - 1);
        // the smooth eigenfunction has zero normal derivative and exact half-bienergy pi^4/2
        EXPECT_LT(relative, previous);
        previous = relative;
    }
    // the finest grid reproduces the continuum Neumann oracle within the historical discretization error
    EXPECT_LT(previous, .002);
}
}   // namespace

namespace {
/// @brief changes rho between evaluations while retaining spatial plans and matrix coefficients
TEST(GeometricSmoothingFEM, MutableRhoPlans) {
    const auto mesh = fdapde::Triangulation<2, 2>::UnitSquare(3);
    using G = fdapde::manifold::CheegerLogEuclideanSPDGeometry<double, 3, fdapde::Usage::InterpolationNodes>;
    const G geometry;
    const fdapde::FeSpace scalar_source_space(mesh, fdapde::P1<1>);
    const fdapde::GeometricSpace space(scalar_source_space, geometry);
    fdapde::MatrixBatch<G::Point> nodes(space.n_dofs());
    for (int i = 0; i < space.n_dofs(); ++i) {
        G::Tangent x;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c <= r; ++c)
                x(r, c) = r == c ? .1 * (r + 1) * (1 + mesh.nodes()(i, 0)) : .08 * std::sin(i + r + c);
        nodes[i] = fdapde::matrix_exp(x);
    }
    fdapde::GeometricFunction field(space, nodes);
    fdapde::MatrixBatch<fdapde::Vector<double, 2>> points(2);
    points[0] = fdapde::Vector<double, 2>(.2, .3);
    points[1] = fdapde::Vector<double, 2>(.65, .7);
    const auto plan = space.prepare_evaluation(points);
    const auto initial = plan(field);
    // visiting interior sites prepares metric-dependent cell data
    EXPECT_GT(field.prepared_cells(), 0u);
    std::vector<double> rho(space.n_dofs());
    for (int i = 0; i < space.n_dofs(); ++i) rho[i] = .4 + .03 * i;
    const auto* cached_log = field.coeff()[0].cache().data();
    field.set_rho(rho);
    // replacing rho drops old cell lifts before a plan can reuse them
    EXPECT_EQ(field.prepared_cells(), 0u);
    // rho-only replacement preserves the original matrix cache buffer
    EXPECT_EQ(field.coeff()[0].cache().data(), cached_log);
    const auto updated = plan(field);
    const fdapde::FeSpace scalar_source_reference_space(mesh, fdapde::P1<1>);
    const fdapde::GeometricSpace reference_space(scalar_source_reference_space, geometry, std::span<const double>(rho));
    const fdapde::GeometricFunction reference(reference_space, nodes);
    const auto expected = reference.eval_at(points);
    double change = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        // the retained spatial plan agrees with a new field built using the replacement metric
        EXPECT_LT((fdapde::Matrix<double, 3, 3>(updated[i] - expected[i]).norm()), 1e-10);
        change += fdapde::Matrix<double, 3, 3>(updated[i] - initial[i]).norm();
    }
    // noncommuting data expose a material value change when the local rotational penalty changes
    EXPECT_GT(change, 1e-6);
    auto bad = rho;
    bad[0] = 0;
    // public updates reject nonpositive rho before discarding the last valid cache
    EXPECT_THROW(field.set_rho(bad), std::invalid_argument);
    // failed updates preserve the active scalar coefficients
    EXPECT_DOUBLE_EQ(field.rho_coefficients()[0], rho[0]);
    field.set_rho({});
    const auto restored = plan(field);
    // an empty override restores the geometry's constant rho using the same spatial plan
    EXPECT_LT((fdapde::Matrix<double, 3, 3>(restored[0] - initial[0]).norm()), 1e-10);
}
}   // namespace

namespace {
/// @brief recovers roundoff-stable boundary support from the TSPDE physical-point regression
TEST(GeometricSmoothingFEM, BarycentricBoundarySupport) {
    const auto mesh = Triangulation<2, 2>::UnitSquare(11);
    const FeSpace space(mesh, P1<1>);
    Eigen::MatrixXd locations(2, 2);
    locations << 493. / 499., 6. / 499., .123, .234;
    const auto cells = mesh.locate(locations);
    const auto edge = gfe::p1_fem_barycentric_weights(space, cells[0], locations.row(0).transpose());
    // the captured point x+y=1 has exactly two active shape functions despite inverse-map roundoff
    EXPECT_EQ(std::count(edge.begin(), edge.end(), 0.), 1);
    const auto interior = gfe::p1_fem_barycentric_weights(space, cells[1], locations.row(1).transpose());
    // interior support is preserved instead of being projected onto a boundary
    EXPECT_GT(*std::min_element(interior.begin(), interior.end()), 0);
    // normalization preserves the P1 partition of unity
    EXPECT_NEAR(std::accumulate(interior.begin(), interior.end(), 0.), 1, 1e-15);
    // a genuinely outside point is rejected rather than clamped onto the cell
    EXPECT_THROW(gfe::p1_fem_barycentric_weights(space, cells[0], Eigen::Vector2d(-1., -1.)), std::domain_error);
    // cell indices are checked before asking the DOF handler for its geometry
    EXPECT_THROW(gfe::p1_fem_barycentric_weights(space, mesh.n_cells(), Eigen::Vector2d(0., 0.)), std::out_of_range);
}
}   // namespace

namespace {
namespace penalty_consistency {
using EMat = Eigen::Matrix3d;
using Six = Eigen::Matrix<double, 6, 1>;
using Metric = Eigen::Matrix<double, 6, 6>;

/** @brief retain one scalar value, gradient and spatial Laplacian */
struct ScalarJet {
    double value = 0, dx = 0, dy = 0, lap = 0;
};
/** @brief multiply scalar jets with the two-dimensional product rule */
ScalarJet operator*(const ScalarJet& a, const ScalarJet& b) {
    return {
      a.value * b.value, a.dx * b.value + a.value * b.dx, a.dy * b.value + a.value * b.dy,
      a.lap * b.value + a.value * b.lap + 2 * (a.dx * b.dx + a.dy * b.dy)};
}
/** @brief add independent scalar jets coefficientwise */
ScalarJet operator+(const ScalarJet& a, const ScalarJet& b) {
    return {a.value + b.value, a.dx + b.dx, a.dy + b.dy, a.lap + b.lap};
}
/** @brief differentiate a scalar composition through its first two derivatives */
ScalarJet compose(const ScalarJet& a, double value, double first, double second) {
    return {value, first * a.dx, first * a.dy, first * a.lap + second * (a.dx * a.dx + a.dy * a.dy)};
}
/** @brief apply the sine composition to a spatial jet */
ScalarJet sine(const ScalarJet& a) { return compose(a, std::sin(a.value), std::cos(a.value), -std::sin(a.value)); }
/** @brief apply the cosine composition to a spatial jet */
ScalarJet cosine(const ScalarJet& a) { return compose(a, std::cos(a.value), -std::sin(a.value), -std::cos(a.value)); }
/** @brief apply the exponential composition to a spatial jet */
ScalarJet exponential(const ScalarJet& a) {
    const double value = std::exp(a.value);
    return compose(a, value, value, value);
}
/** @brief store matrix values and their two-dimensional spatial derivatives */
struct MatrixJet {
    EMat value = EMat::Zero(), dx = EMat::Zero(), dy = EMat::Zero(), lap = EMat::Zero();
};
/** @brief multiply matrix jets without commuting matrix factors */
MatrixJet operator*(const MatrixJet& a, const MatrixJet& b) {
    return {
      a.value * b.value, a.dx * b.value + a.value * b.dx, a.dy * b.value + a.value * b.dy,
      a.lap * b.value + a.value * b.lap + 2 * (a.dx * b.dx + a.dy * b.dy)};
}
/** @brief transpose every derivative of a matrix jet */
MatrixJet transpose(const MatrixJet& a) {
    return {a.value.transpose(), a.dx.transpose(), a.dy.transpose(), a.lap.transpose()};
}
/** @brief assign a scalar jet to one matrix entry */
void set(MatrixJet& m, int i, int j, const ScalarJet& a) {
    m.value(i, j) = a.value;
    m.dx(i, j) = a.dx;
    m.dy(i, j) = a.dy;
    m.lap(i, j) = a.lap;
}
/** @brief form a rotation about one coordinate axis with analytic spatial derivatives */
MatrixJet rotation(const ScalarJet& angle, int axis) {
    MatrixJet out;
    const int first = (axis + 1) % 3, second = (axis + 2) % 3;
    const auto c = cosine(angle), s = sine(angle);
    out.value(axis, axis) = 1;
    set(out, first, first, c);
    set(out, second, second, c);
    set(out, second, first, s);
    set(out, first, second, ScalarJet {-1} * s);
    return out;
}
/** @brief reproduce the canonical noncommuting cosine profiles on the unit square */
std::array<ScalarJet, 3> field(double x, double y, bool mixed) {
    const double pi = std::acos(-1.);
    const auto profile = [&](double offset, double amplitude, int nx, int ny) {
        const double a = nx * pi, b = ny * pi, q = amplitude * std::cos(a * x) * std::cos(b * y);
        return ScalarJet {
          offset + q, -amplitude * a * std::sin(a * x) * std::cos(b * y),
          -amplitude * b * std::cos(a * x) * std::sin(b * y), -(a * a + b * b) * q};
    };
    return {profile(0., mixed ? .12 : 0., 1, 1), profile(.6, mixed ? .15 : 0., 1, 0), profile(.2, .3, 1, 1)};
}
/** @brief retain the SPD2 determinant and eigenvalue ratio with a repeated minor pair */
std::pair<MatrixJet, MatrixJet> tensor_field(double x, double y, bool mixed) {
    const auto [s, a, t] = field(x, y, mixed);
    const MatrixJet r = rotation(t, 2);
    MatrixJet log_diagonal, diagonal;
    const std::array<double, 3> shape {4. / 3., -2. / 3., -2. / 3.};
    for (int i = 0; i < 3; ++i) {
        const auto ell = ScalarJet {2. / 3.} * s + ScalarJet {shape[i]} * a;
        set(log_diagonal, i, i, ell);
        set(diagonal, i, i, exponential(ell));
    }
    return {r * log_diagonal * transpose(r), r * diagonal * transpose(r)};
}
/** @brief flatten symmetric matrix values in the same six-coordinate order as the drivers */
Six raw(const EMat& m) {
    Six out;
    out << m(0, 0), m(0, 1), m(0, 2), m(1, 1), m(1, 2), m(2, 2);
    return out;
}
/** @brief restore symmetric matrices from six raw coordinate values */
EMat matrix(const Six& v) {
    EMat out;
    out << v[0], v[1], v[2], v[1], v[3], v[4], v[2], v[4], v[5];
    return out;
}
/** @brief form one raw symmetric-coordinate basis for the independent metric oracle */
EMat basis(int a) {
    Six v = Six::Zero();
    v[a] = 1;
    return matrix(v);
}
/** @brief pair symmetric matrices in the full Frobenius metric */
double inner(const EMat& a, const EMat& b) { return (a.array() * b.array()).sum(); }
/** @brief evaluate the commutator without assuming that its factors commute */
EMat commutator(const EMat& a, const EMat& b) { return a * b - b * a; }
/** @brief differentiate the metric operator analytically in one log-matrix direction */
EMat operator_derivative(const EMat& x, const EMat& u, const EMat& v, double rho) {
    return (commutator(u, commutator(x, v)) + commutator(x, commutator(u, v))) / rho;
}
/** @brief retain the independent C-LE metric and its exact Christoffel contractions */
struct CheegerMetric {
    EMat x;
    double rho;
    Metric inverse_operator, metric;
    /** @brief invert the logarithmic commutator operator in six raw coordinates */
    CheegerMetric(const EMat& x_, double rho_) : x(x_), rho(rho_) {
        Metric operator_matrix;
        for (int a = 0; a < 6; ++a) {
            const EMat u = basis(a);
            operator_matrix.col(a) = raw(u + commutator(x, commutator(x, u)) / rho);
        }
        inverse_operator = operator_matrix.inverse();
        metric = inverse_operator;
        for (int a : {1, 2, 4}) metric.row(a) *= 2;
    }
    /** @brief apply the C-LE metric to one symmetric tangent */
    EMat apply(const EMat& v) const { return matrix(inverse_operator * raw(v)); }
    /** @brief contract the Christoffel tensor against one spatial log derivative */
    EMat connection(const EMat& v) const {
        const EMat mv = apply(v);
        Six dual;
        for (int a = 0; a < 6; ++a) {
            const EMat e = basis(a), me = matrix(inverse_operator.col(a));
            dual[a] =
              -inner(me, operator_derivative(x, v, mv, rho)) + .5 * inner(mv, operator_derivative(x, e, mv, rho));
        }
        return matrix(metric.ldlt().solve(dual));
    }
};
/** @brief solve the independent SPD Sylvester equation using its eigenframe */
EMat lyapunov(const EMat& p, const EMat& v) {
    Eigen::SelfAdjointEigenSolver<EMat> spectrum(p);
    const EMat rotated = spectrum.eigenvectors().transpose() * v * spectrum.eigenvectors();
    EMat result;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            result(i, j) = rotated(i, j) / (spectrum.eigenvalues()[i] + spectrum.eigenvalues()[j]);
    return spectrum.eigenvectors() * result * spectrum.eigenvectors().transpose();
}
/** @brief differentiate Cholesky factors independently from P=LL transpose */
EMat cholesky_derivative(const EMat& l, const EMat& v) {
    const EMat inverse = l.inverse(), transformed = inverse * v * inverse.transpose();
    EMat lower = EMat::Zero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j) lower(i, j) = transformed(i, j) * (i == j ? .5 : 1.);
    return l * lower;
}
/** @brief evaluate independent continuum half-tension energy in the selected geometries */
std::array<double, 6> density(double x, double y, bool mixed, bool include_cheeger = true) {
    const auto [log, p] = tensor_field(x, y, mixed);
    const EMat inverse = p.value.inverse();
    const EMat airm = p.lap - p.dx * inverse * p.dx - p.dy * inverse * p.dy;
    const double rho = .25;
    double cheeger_energy = 0;
    if (include_cheeger) {
        const CheegerMetric metric(log.value, rho);
        const EMat ce = log.lap + metric.connection(log.dx) + metric.connection(log.dy);
        cheeger_energy = .5 * inner(ce, metric.apply(ce));
    }
    const EMat lx = lyapunov(p.value, p.dx), ly = lyapunov(p.value, p.dy);
    const EMat bw = p.lap - 2 * (lx * p.value * lx + ly * p.value * ly);
    const EMat l = Eigen::LLT<EMat>(p.value).matrixL();
    const EMat ldx = cholesky_derivative(l, p.dx), ldy = cholesky_derivative(l, p.dy);
    EMat lc = cholesky_derivative(l, p.lap - 2 * (ldx * ldx.transpose() + ldy * ldy.transpose()));
    for (int i = 0; i < 3; ++i)
        lc(i, i) = lc(i, i) / l(i, i) - (ldx(i, i) * ldx(i, i) + ldy(i, i) * ldy(i, i)) / (l(i, i) * l(i, i));
    double lc_energy = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j) lc_energy += .5 * lc(i, j) * lc(i, j);
    return {
      .5 * p.lap.squaredNorm(),
      .5 * log.lap.squaredNorm(),
      .5 * (inverse * airm * inverse * airm).trace(),
      cheeger_energy,
      .25 * inner(bw, lyapunov(p.value, bw)),
      lc_energy};
}
/** @brief integrate continuum densities with deterministic midpoint reductions */
std::array<double, 6> oracle(int side, bool mixed, bool include_cheeger = true) {
    std::array<double, 6> out {};
    constexpr int block_rows = 16;
    std::vector<std::array<double, 6>> samples(std::min(side, block_rows) * side);
    for (int first = 0; first < side; first += block_rows) {
        const std::size_t count = std::min(block_rows, side - first) * side;
        const auto evaluate = [&](std::size_t index) {
            samples[index] =
              density((first + index / side + .5) / side, (index % side + .5) / side, mixed, include_cheeger);
        };
        if (count >= 32 && fdapde::parallel_get_num_threads() > 1)
            fdapde::gfe::internals::for_each_point(count, fdapde::execution_par, evaluate);
        else
            fdapde::gfe::internals::for_each_point(count, fdapde::execution_seq, evaluate);
        for (std::size_t index = 0; index < count; ++index)
            for (int g = 0; g < 6; ++g) out[g] += samples[index][g] / (side * side);
    }
    return out;
}
}   // namespace penalty_consistency
// noncommuting Neumann fields exercise every SPD3 tension against an independent analytic continuum oracle
TEST(GeometricSmoothingFEM, NoncommutingSPD3ContinuumRefinement) {
    using namespace penalty_consistency;
    using Point = SPDMatrix<
      double, 3,
      Cache::Union<
        Cache::Log, Cache::Spectral, Cache::LogDividedDifferences, Cache::Sqrt, Cache::InverseSqrt, Cache::Cholesky,
        Cache::LogCholesky>>;
    for (bool mixed : {false, true}) {
        const auto exact = oracle(32, mixed), refined = oracle(64, mixed);
        for (int metric = 0; metric < 6; ++metric) {
            // independent midpoint refinement must stabilize before it is used as the discretization oracle
            ASSERT_LT(std::abs(exact[metric] / refined[metric] - 1), 1e-9);
        }
        std::array<double, 6> previous;
        previous.fill(std::numeric_limits<double>::infinity());
        for (int side : {9, 17, 33, 65}) {
            const auto mesh = Triangulation<2, 2>::UnitSquare(side);
            const FeSpace space(mesh, P1<1>);
            const auto stencil = gfe::p1_lumped_laplacian_stencil(space, QS2DP4);
            const gfe::PreparedP1LumpedLaplacianStencil prepared(stencil);
            MatrixBatch<Point> nodes(space.n_dofs());
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                const auto x = mesh.node(i);
                const EMat p = tensor_field(x[0], x[1], mixed).second.value;
                Matrix<double, 3, 3> value;
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col) value(row, col) = p(row, col);
                nodes[i] = value;
            }
            const EMat a = tensor_field(0, 0, mixed).second.value;
            const EMat b = tensor_field(1, 0, mixed).second.value;
            // the endpoint commutator excludes accidental reduction to a common eigenframe
            ASSERT_GT((a * b - b * a).norm(), .1);
            std::vector<EMat> residual(nodes.size(), EMat::Zero());
            for (const auto& edge : stencil.edges) {
                const auto x = mesh.node(edge.first), y = mesh.node(edge.second);
                const EMat delta = edge.stiffness * (tensor_field(y[0], y[1], mixed).second.value -
                                                     tensor_field(x[0], x[1], mixed).second.value);
                residual[edge.first] += delta;
                residual[edge.second] -= delta;
            }
            double euclidean = 0;
            for (std::size_t i = 0; i < nodes.size(); ++i)
                euclidean += .5 * residual[i].squaredNorm() / stencil.lumped_masses[i];
            const std::array<double, 6> energy {
              euclidean,
              gfe::p1_discrete_tension_value(manifold::LogEuclideanSPDGeometry<double, 3> {}, nodes, prepared).value,
              gfe::p1_discrete_tension_value(manifold::AffineInvariantSPDGeometry<double, 3> {}, nodes, prepared).value,
              gfe::p1_cheeger_discrete_tension_log_contribution(
                manifold::CheegerLogEuclideanSPDGeometry<double, 3>(.5), nodes, prepared)
                .value,
              gfe::p1_discrete_tension_value(manifold::BuresWassersteinSPDGeometry<double, 3> {}, nodes, prepared)
                .value,
              gfe::p1_discrete_tension_value(manifold::LogCholeskySPDGeometry<double, 3> {}, nodes, prepared).value};
            for (int metric = 0; metric < 6; ++metric) {
                SCOPED_TRACE(::testing::Message() << "mixed=" << mixed << " side=" << side << " metric=" << metric);
                const double error = std::abs(energy[metric] / refined[metric] - 1);
                // each refined FEM energy must approach its own independent metric-specific continuum value
                EXPECT_LT(error, previous[metric]);
                previous[metric] = error;
            }
        }
        for (double error : previous) {
            // the 65-by-65 mesh retains the original two-percent continuum consistency threshold
            EXPECT_LT(error, .02);
        }
    }
}
}   // namespace
