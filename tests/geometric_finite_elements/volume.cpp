// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/geometric_finite_elements_fem.h>
#include <gtest/gtest.h>

#include <chrono>

namespace {
using namespace fdapde;
using Sym = SymmetricMatrix<double, 3>;
using Point = SPDMatrix<double, 3>;
using LE = manifold::LogEuclideanSPDGeometry<double, 3, Usage::InterpolationNodes>;
using LC = manifold::LogCholeskySPDGeometry<double, 3, Usage::InterpolationNodes>;
using AI = manifold::AffineInvariantSPDGeometry<double, 3, Usage::BasePointMaps>;
using BW = manifold::BuresWassersteinSPDGeometry<double, 3, Usage::BasePointMaps>;
using CE = manifold::CheegerLogEuclideanSPDGeometry<double, 3, Usage::InterpolationNodes>;

/// @brief makes a right tetrahedron with independent unit coordinate gradients
Triangulation<3, 3> tetrahedron() {
    Eigen::Matrix<double, 4, 3> vertices;
    vertices << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    Eigen::Matrix<int, 1, 4> cells;
    cells << 0, 1, 2, 3;
    return Triangulation<3, 3>(vertices, cells, Eigen::Matrix<int, 4, 1>::Ones());
}

/// @brief makes a smooth noncommuting logarithmic field depending on all spatial coordinates
Sym log_field(const Eigen::Vector3d& x) {
    Sym q;
    q(0, 0) = .2 + .08 * x[0];
    q(1, 1) = .45 + .06 * x[1];
    q(2, 2) = .7 - .04 * x[2];
    q(1, 0) = .03 + .04 * x[2];
    q(2, 0) = .02 - .03 * x[1];
    q(2, 1) = -.02 + .05 * x[0];
    return q;
}

/// @brief supplies the affine coefficient chart to the existing native descent solver
struct CoefficientChart {
    using Point = Eigen::VectorXd;
    using Tangent = Eigen::VectorXd;
    /// @brief reports six independent coefficients at each of four tetrahedral vertices
    std::size_t dimension() const { return 24; }
    /// @brief pairs coefficient directions in the fixed chart
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u.dot(v); }
    /// @brief measures a coefficient direction
    double norm(const Point&, const Tangent& u) const { return u.norm(); }
    /// @brief retains all symmetric coordinate directions
    Tangent project(const Point&, const Tangent& u) const { return u; }
    /// @brief constructs the zero displacement at the supplied coefficient vector
    Tangent zero_tangent(const Point& x) const { return Tangent::Zero(x.size()); }
    /// @brief combines coefficient directions without changing their chart
    Tangent linear_combination(const Point&, double a, const Tangent& u, double b, const Tangent& v) const {
        return a * u + b * v;
    }
    /// @brief takes an affine coefficient step
    Point retract(const Point& x, const Tangent& u, double step) const { return x + step * u; }
};

/// @brief evaluates nodal Frobenius fidelity plus the assembled volumetric discrete tension
/// @details intrinsic models use logarithmic coordinates and Euclidean smoothing uses ambient coordinates
template <typename Geometry, bool Euclidean = false> struct VolumeProblem {
    /// @brief retains a cost and analytic gradient at one solver-owned candidate
    struct Workspace {
        std::optional<std::pair<double, Eigen::VectorXd>> value;
    };
    Geometry geometry;
    gfe::P1LumpedLaplacianStencil stencil;
    MatrixBatch<typename Geometry::Point> observations;
    static constexpr double lambda = .005;

    /// @brief binds the model and independently assembled tetrahedral stencil
    VolumeProblem(const Geometry& g, const gfe::P1LumpedLaplacianStencil& s, const Triangulation<3, 3>& mesh) :
        geometry(g), stencil(s), observations(4) {
        for (int i = 0; i < 4; ++i) observations[i] = matrix_exp(log_field(mesh.node(i)));
    }
    /// @brief unpacks the six independent entries at one node
    static Sym unpack(const Eigen::VectorXd& x, int node) {
        Sym q;
        int a = 6 * node;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c <= r; ++c) q(r, c) = x[a++];
        return q;
    }
    /// @brief packs the observation field into its model's coefficient chart
    Eigen::VectorXd initial() const {
        Eigen::VectorXd x(24);
        for (int i = 0; i < 4; ++i) {
            const Sym q = Euclidean ? Sym(observations[i]) : Sym(matrix_log(observations[i]));
            int a = 6 * i;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c <= r; ++c) x[a++] = q(r, c);
        }
        return x;
    }
    /// @brief reconstructs nodal tensors while retaining each geometry's native owner caches
    auto nodes(const Eigen::VectorXd& x) const {
        MatrixBatch<typename Geometry::Point> batch(4);
        for (int i = 0; i < 4; ++i) {
            if constexpr (Euclidean)
                batch[i] = unpack(x, i);
            else
                batch[i] = matrix_exp(unpack(x, i));
        }
        return batch;
    }
    /// @brief evaluates either ordered serial kernels or their execution counterpart
    template <typename Policy> std::pair<double, Eigen::VectorXd> evaluate(const Eigen::VectorXd& x, Policy policy) {
        const auto batch = nodes(x);
        std::vector<Sym> metric_gradient(4, Sym {});
        double value = 0;
        if constexpr (Euclidean) {
            for (const auto& edge : stencil.edges) {
                const Sym increment(batch[edge.second] - batch[edge.first]);
                metric_gradient[edge.first] += edge.stiffness * increment;
                metric_gradient[edge.second] -= edge.stiffness * increment;
            }
            const auto residual = metric_gradient;
            for (int i = 0; i < 4; ++i) {
                value += .5 * manifold::internals::cheeger_inner(residual[i], residual[i]) / stencil.lumped_masses[i];
                metric_gradient[i] = Sym {};
            }
            for (const auto& edge : stencil.edges) {
                const Sym increment(
                  residual[edge.second] / stencil.lumped_masses[edge.second] -
                  residual[edge.first] / stencil.lumped_masses[edge.first]);
                metric_gradient[edge.first] += edge.stiffness * increment;
                metric_gradient[edge.second] -= edge.stiffness * increment;
            }
        } else if constexpr (std::same_as<Geometry, CE>) {
            const auto result = gfe::p1_cheeger_discrete_tension_log_contribution(geometry, batch, stencil, policy);
            value = result.value;
            metric_gradient = result.nodal_gradient;
        } else {
            const auto result = gfe::p1_discrete_tension_contribution(geometry, batch, stencil, policy);
            value = result.value;
            metric_gradient = result.nodal_gradient;
        }
        value *= lambda;
        Eigen::VectorXd gradient(24);
        for (int i = 0; i < 4; ++i) {
            const Sym residual(batch[i] - observations[i]);
            value += .5 * manifold::internals::cheeger_inner(residual, residual);
            int a = 6 * i;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c <= r; ++c) {
                    Sym basis = Sym {};
                    basis(r, c) = 1;
                    const Sym direction = Euclidean ? basis : Sym(matrix_exp_frechet(unpack(x, i), basis));
                    const double fidelity = manifold::internals::cheeger_inner(residual, direction);
                    double penalty;
                    if constexpr (Euclidean || std::same_as<Geometry, CE>)
                        penalty = manifold::internals::cheeger_inner(metric_gradient[i], basis);
                    else
                        penalty = geometry.inner_product(batch[i], metric_gradient[i], direction);
                    gradient[a++] = fidelity + lambda * penalty;
                }
        }
        return {value, gradient};
    }
    /// @brief caches the exact serial cost and analytic gradient for native optimization
    std::pair<double, Eigen::VectorXd> cost_gradient(const Eigen::VectorXd& x, Workspace& workspace) {
        if (!workspace.value) workspace.value = evaluate(x, execution_seq);
        return *workspace.value;
    }
    /// @brief exposes the cached scalar objective to Armijo backtracking
    double cost(const Eigen::VectorXd& x, Workspace& workspace) { return cost_gradient(x, workspace).first; }
    /// @brief exposes the cached coefficient covector to the descent solver
    Eigen::VectorXd grad(const Eigen::VectorXd& x, Workspace& workspace) { return cost_gradient(x, workspace).second; }
};

/// @brief checks tetrahedral interpolation, independent objective derivatives and native smoothing descent
template <typename Geometry, bool Euclidean = false>
void check_volume(const Geometry& geometry, const std::string& model) {
    SCOPED_TRACE(model);
    const auto mesh = tetrahedron();
    const GeometricFeSpace space(mesh, P1<1>, geometry);
    const auto packet = gfe::p1_fem_cell_quadrature(space, 0, QS3DP6);
    const auto stencil = gfe::p1_lumped_laplacian_stencil(space, QS3DP6);
    VolumeProblem<Geometry, Euclidean> problem(geometry, stencil, mesh);
    auto x = problem.initial();
    const auto nodes = problem.nodes(x);
    const Sym first(nodes[0]), second(nodes[1]);
    // the varying field exercises noncommuting matrix paths rather than a scalar commuting reduction
    EXPECT_GT((Matrix<double, 3, 3>(first * second - second * first).norm()), 1e-4);
    const Eigen::Vector3d location(.2, .3, .1);
    const auto weights = gfe::p1_fem_barycentric_weights(space, 0, location);
    if constexpr (Euclidean) {
        Sym actual = Sym {}, expected = Sym {};
        for (int i = 0; i < 4; ++i) {
            const Sym node(nodes[i]);
            actual += weights[i] * node;
            expected += std::array {.4, .2, .3, .1}[i] * node;
        }
        // arithmetic interpolation uses the independent physical tetrahedron barycentric coordinates
        EXPECT_LT((Matrix<double, 3, 3>(actual - expected).norm()), 1e-14);
    } else {
        const GeometricFeFunction field(space, nodes);
        const Point actual(field(location));
        const auto reference = gfe::p1_geodesic_value(geometry, nodes, std::array {.4, .2, .3, .1});
        // the physical-space evaluation and independent four-node mean both have valid stationary branches
        ASSERT_TRUE(reference.converged());
        // tetrahedral location and DOF ordering reproduce the independently weighted mean
        EXPECT_LT((Matrix<double, 3, 3>(actual - reference.value).norm()), 1e-9);
        for (int i = 0; i < 4; ++i) {
            const Point vertex(field(mesh.node(i)));
            // each volume vertex reproduces its own nonconstant tensor exactly
            EXPECT_LT((Matrix<double, 3, 3>(vertex - nodes[i]).norm()), 1e-13);
        }
        const auto linearization = gfe::p1_geodesic_linearization(geometry, nodes, packet.barycentric_weights[0]);
        const auto derivative = linearization.weight_jvp(packet.physical_weight_gradients[2]);
        const Sym velocity = [&] {
            if constexpr (requires { derivative.derivative; }) {
                // implicit differential branches must retain their original solve certificate
                EXPECT_TRUE(derivative.converged());
                return Sym(derivative.derivative);
            } else {
                return Sym(derivative);
            }
        }();
        auto wp = packet.barycentric_weights[0], wm = wp;
        for (int i = 0; i < 4; ++i) {
            wp[i] += 1e-5 * packet.physical_weight_gradients[2][i];
            wm[i] -= 1e-5 * packet.physical_weight_gradients[2][i];
        }
        const auto p = gfe::p1_geodesic_value(geometry, nodes, wp);
        const auto m = gfe::p1_geodesic_value(geometry, nodes, wm);
        // independently refitted third-axis perturbations remain stationary on the same interior branch
        ASSERT_TRUE(p.converged() && m.converged());
        // the tetrahedral z-gradient matches a centered spatial difference for the four-node mean
        EXPECT_LT((Matrix<double, 3, 3>(velocity - (p.value - m.value) / 2e-5).norm()), 2e-6);
        std::vector<Sym> log_directions(4), directions(4);
        auto plus = nodes, minus = nodes;
        for (int i = 0; i < 4; ++i) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c <= r; ++c) log_directions[i](r, c) = .08 * std::cos(i + r + 2 * c);
            directions[i] = matrix_exp_frechet(Sym(matrix_log(nodes[i])), log_directions[i]);
            if constexpr (std::same_as<Geometry, CE>) {
                const Sym chart(matrix_log(nodes[i]));
                const Sym log_plus(chart + 1e-5 * log_directions[i]);
                const Sym log_minus(chart - 1e-5 * log_directions[i]);
                plus[i] = matrix_exp(log_plus);
                minus[i] = matrix_exp(log_minus);
            } else {
                plus[i] = geometry.exponential(nodes[i], directions[i], 1e-5);
                minus[i] = geometry.exponential(nodes[i], directions[i], -1e-5);
            }
        }
        const Sym observation(nodes[0]);
        auto contribution = [&](const auto& batch) {
            if constexpr (std::same_as<Geometry, CE>)
                return gfe::p1_cheeger_frobenius_data_site_log_contribution(geometry, batch, weights, observation);
            else
                return gfe::p1_frobenius_data_site_contribution(geometry, batch, weights, observation);
        };
        const auto data = contribution(nodes), dp = contribution(plus), dm = contribution(minus);
        // four-node mean and pullback solves must converge before testing the sparse observation gradient
        ASSERT_TRUE(data.converged() && dp.converged() && dm.converged());
        double analytic = 0;
        for (int i = 0; i < 4; ++i) {
            if constexpr (std::same_as<Geometry, CE>)
                analytic += manifold::internals::cheeger_inner(data.nodal_gradient[i], log_directions[i]);
            else
                analytic += geometry.inner_product(nodes[i], data.nodal_gradient[i], directions[i]);
        }
        // refitted tetrahedral means independently validate the actual sparse Frobenius data pullback
        EXPECT_NEAR(analytic, (dp.value - dm.value) / 2e-5, 3e-6);
    }
    const auto serial = problem.evaluate(x, execution_seq), parallel = problem.evaluate(x, execution_par);
    // the volumetric assembled objective preserves ordered scalar reduction under four-worker execution
    EXPECT_EQ(serial.first, parallel.first);
    // the coefficient pullback preserves every entry when pair kernels are scheduled in parallel
    EXPECT_EQ((serial.second - parallel.second).norm(), 0);
    Eigen::VectorXd direction(24);
    for (int i = 0; i < 24; ++i) direction[i] = .1 * std::cos(i + 1);
    constexpr double h = 1e-5;
    const double plus = problem.evaluate(x + h * direction, execution_seq).first;
    const double minus = problem.evaluate(x - h * direction, execution_seq).first;
    // centered coordinate perturbations independently validate both fidelity and discrete-tension gradients
    EXPECT_NEAR(serial.second.dot(direction), (plus - minus) / (2 * h), 3e-6);
    manifold::SteepestDescentOptions options;
    options.max_iterations = 100;
    const auto start = std::chrono::steady_clock::now();
    const auto result = manifold::RiemannianSteepestDescent(options).optimize(problem, CoefficientChart {}, x);
    ::testing::Test::RecordProperty(
      model + "_fit_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    ::testing::Test::RecordProperty(model + "_fit_iterations", int(result.iterations));
    ::testing::Test::RecordProperty(model + "_gradient_norm", result.gradient_norm);
    ::testing::Test::RecordProperty(model + "_initial_cost", serial.first);
    ::testing::Test::RecordProperty(model + "_final_cost", result.cost);
    // native Armijo descent must move away from the nonsmoothed varying observation field
    EXPECT_GT(result.iterations, 0u);
    // accepted smoothing steps reduce the unchanged nodal fidelity plus volumetric penalty
    EXPECT_LT(result.cost, serial.first);
    // the final iterate retains a finite objective certificate after all accepted or rejected trials
    EXPECT_TRUE(std::isfinite(result.gradient_norm));
    // default first-order tolerance remains the stopping certificate of the small smoothing fit
    EXPECT_TRUE(result.converged());
    if constexpr (Euclidean) {
        Eigen::Matrix4d stiffness;
        stiffness << 3, -1, -1, -1, -1, 1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 1;
        stiffness /= 6.;
        const Eigen::Matrix4d hessian =
          Eigen::Matrix4d::Identity() + problem.lambda * 24 * stiffness.transpose() * stiffness;
        const auto factor = hessian.ldlt();
        for (int a = 0; a < 6; ++a) {
            Eigen::Vector4d observed, fitted;
            for (int i = 0; i < 4; ++i) {
                observed[i] = x[6 * i + a];
                fitted[i] = result.point[6 * i + a];
            }
            // the native descent fit agrees with the independent exact Euclidean normal equations
            EXPECT_LT((fitted - factor.solve(observed)).norm(), 2e-6);
        }
    }
}

/// @brief catches face-sized filters and boundary masks on a cube whose faces outnumber edges
TEST(VolumetricGeometry, FaceIterationAndBoundaryMask) {
    auto mesh = Triangulation<3, 3>::UnitCube(3);
    // independent topology counts distinguish the face and edge filter lengths on the regression mesh
    ASSERT_EQ(mesh.n_faces(), 120);
    // the six-tetrahedron cube subdivision has 98 unique edges
    ASSERT_EQ(mesh.n_edges(), 98);
    int visited = 0;
    for (auto it = mesh.faces_begin(); it != mesh.faces_end(); ++it) {
        // unfiltered traversal visits faces in their native contiguous enumeration
        EXPECT_EQ(it->id(), visited++);
    }
    // every face remains reachable after traversal crosses the edge-count boundary
    EXPECT_EQ(visited, mesh.n_faces());
    Vector<bool, Dynamic> mask = Vector<bool, Dynamic>::Zero(mesh.n_faces());
    int selected = -1;
    for (auto it = mesh.boundary_faces_begin(); it != mesh.boundary_faces_end(); ++it)
        if (it->id() >= mesh.n_edges()) selected = it->id();
    // the regression requires a genuine boundary face beyond the erroneous old filter length
    ASSERT_GE(selected, mesh.n_edges());
    mask[selected] = true;
    // a correctly face-sized public mask is accepted rather than checked against the edge count
    ASSERT_NO_THROW(mesh.mark_boundary(mask));
    int marked = 0;
    for (auto it = mesh.boundary_begin(1); it != mesh.boundary_end(1); ++it) {
        // the selected face alone carries the public boundary marker
        EXPECT_EQ(it->id(), selected);
        ++marked;
    }
    // exactly one boundary face is selected by the full-length boolean mask
    EXPECT_EQ(marked, 1);
    // the public overload accepts a lazy boolean expression through its derived coefficient accessor
    EXPECT_NO_THROW(mesh.mark_boundary(mask | mask));
    const Matrix<bool, Dynamic, Dynamic> wide = Matrix<bool, Dynamic, Dynamic>::Zero(mesh.n_faces(), 2);
    // a face count alone is insufficient because the public mask represents one boolean per face
    EXPECT_THROW(mesh.mark_boundary(wide), std::invalid_argument);
    const Vector<bool, Dynamic> wrong = Vector<bool, Dynamic>::Zero(mesh.n_edges());
    // debug API validation rejects an edge-sized mask before indexing boundary faces
    EXPECT_THROW(mesh.mark_boundary(wrong), std::invalid_argument);
}

/// @brief verifies positive interior quadrature and exact P1 mass and stiffness on a reference tetrahedron
TEST(VolumetricSmoothing, PositiveQuadratureAndAssembly) {
    const auto mesh = tetrahedron();
    const FeSpace space(mesh, P1<1>);
    const auto packet = gfe::p1_fem_cell_quadrature(space, 0, QS3DP6);
    const auto stencil = gfe::p1_lumped_laplacian_stencil(space, QS3DP6);
    for (const auto& weights : packet.barycentric_weights) {
        // strictly interior sites satisfy the differentiability precondition for C-LE weight directions
        EXPECT_GT(*std::min_element(weights.begin(), weights.end()), 0);
    }
    for (double weight : packet.integration_weights) {
        // each geometric quadrature contribution has a strictly positive physical weight
        EXPECT_GT(weight, 0);
    }
    // the reference tetrahedron volume is the determinant-one Jacobian divided by six
    EXPECT_NEAR(
      std::accumulate(packet.integration_weights.begin(), packet.integration_weights.end(), 0.), 1. / 6, 1e-14);
    for (double mass : stencil.lumped_masses) {
        // integrating each barycentric coordinate gives one quarter of the exact tetrahedron volume
        EXPECT_NEAR(mass, 1. / 24, 1e-14);
    }
    // orthogonal coordinate gradients leave exactly the three edges incident to vertex zero nonzero
    ASSERT_EQ(stencil.edges.size(), 3u);
    for (const auto& edge : stencil.edges) {
        // the negative unit gradient dot product integrates to minus the exact tetrahedron volume
        EXPECT_NEAR(edge.stiffness, -1. / 6, 1e-14);
    }
    for (int axis = 0; axis < 3; ++axis)
        for (int node = 0; node < 4; ++node) {
            // all three physical gradient components match the independently differentiated unit basis
            EXPECT_EQ(packet.physical_weight_gradients[axis][node], node == 0 ? -1. : double(node == axis + 1));
        }
    // a negative quadrature weight is rejected by the permanent geometric integration contract
    EXPECT_THROW(gfe::p1_fem_cell_quadrature(space, 0, QS3DP4), std::invalid_argument);
}

/// @brief exercises varying SPD3 fields and volumetric smoothing for every approved geometry
TEST(VolumetricSmoothing, AllSixModels) {
    parallel_set_num_threads(4);
    // the execution lane uses the requested four-worker pool before its first parallel call
    ASSERT_EQ(parallel_get_num_threads(), 4);
    check_volume<LE, true>(LE {}, "Euclidean");
    check_volume(LE {}, "LE");
    check_volume(LC {}, "LC");
    check_volume(AI {}, "AIRM");
    check_volume(BW {}, "BW");
    check_volume(CE {}, "CLE");
}
}   // namespace
