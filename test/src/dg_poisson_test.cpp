#include <Eigen/SparseLU>
#include <fdaPDE/finite_elements.h>

namespace fdapde {

namespace {

struct dg_error {
    double l2;
    double broken_h1;
    int n_dofs;
};

template <typename Space, typename ExactSolution, typename ExactGradient>
dg_error compute_dg_error(
  const Space& space, const Eigen::Matrix<double, Dynamic, 1>& coefficients, const ExactSolution& exact_solution,
  const ExactGradient& exact_gradient) {
    using Quadrature = internals::fe_quadrature_simplex<2, 12>;
    constexpr int n_quadrature_nodes = Quadrature::order;
    double l2_squared = 0;
    double broken_h1_squared = 0;

    for (int cell_id = 0; cell_id < space.triangulation().n_cells(); ++cell_id) {
        auto cell = space.dof_handler().cell(cell_id);
        auto active_dofs = space.dof_handler().active_dofs(cell_id);
        for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
            Vector<double, 2> reference_point = Quadrature::nodes.row(q_k).transpose();
            Eigen::Vector2d point = cell.J() * reference_point.as_eigen_matrix() + cell.node(0);
            double approximate_value = 0;
            Vector<double, 2> approximate_gradient = Vector<double, 2>::Zero();
            for (int i = 0; i < space.n_shape_functions(); ++i) {
                approximate_value += coefficients[active_dofs[i]] * space.eval_shape_value(i, reference_point);
                approximate_gradient +=
                  coefficients[active_dofs[i]] * space.eval_cell_grad(i, cell_id, reference_point);
            }
            const double weight = Quadrature::weights[q_k] * cell.measure();
            l2_squared += weight * std::pow(approximate_value - exact_solution(point), 2);
            const Eigen::Vector2d exact_gradient_at_point = exact_gradient(point);
            double gradient_error_squared = 0;
            for (int d = 0; d < 2; ++d) {
                gradient_error_squared += std::pow(approximate_gradient[d] - exact_gradient_at_point[d], 2);
            }
            broken_h1_squared += weight * gradient_error_squared;
        }
    }
    return {std::sqrt(l2_squared), std::sqrt(broken_h1_squared), space.n_dofs()};
}

template <typename Force, typename ExactSolution, typename ExactGradient>
dg_error solve_dg_poisson(
  int n_nodes, const Force& force, const ExactSolution& exact_solution, const ExactGradient& exact_gradient) {
    auto mesh = Triangulation<2, 2>::UnitSquare(n_nodes);
    FeSpace space(mesh, DG<1, 1>);
    TrialFunction u(space);
    TestFunction v(space);
    auto normal = facet_normal(mesh);
    auto h = facet_size(mesh);
    constexpr double penalty = 10.0;

    auto sipg_form = dot(grad(u), grad(v)) - dot(avg(grad(u)), normal) * jump(v) -
                     dot(avg(grad(v)), normal) * jump(u) + (penalty / h) * jump(u) * jump(v);
    Eigen::SparseMatrix<double> system_matrix = integral(mesh)(sipg_form).assemble();

    ScalarField<2, Force> force_field(force);
    Eigen::Matrix<double, Dynamic, 1> rhs = integral(mesh, QS2DP6)(force_field * v).assemble();

    space.impose_dirichlet_constraint([]([[maybe_unused]] const auto& point) { return 0.0; });
    space.dof_handler().enforce_constraints(system_matrix, rhs);
    system_matrix.makeCompressed();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(system_matrix);
    if (solver.info() != Eigen::Success) return {INFINITY, INFINITY, space.n_dofs()};
    Eigen::Matrix<double, Dynamic, 1> coefficients = solver.solve(rhs);
    if (solver.info() != Eigen::Success) return {INFINITY, INFINITY, space.n_dofs()};
    return compute_dg_error(space, coefficients, exact_solution, exact_gradient);
}

double convergence_rate(double coarse_error, double fine_error) {
    return std::log(coarse_error / fine_error) / std::log(2.0);
}

}   // namespace

TEST(dg_solution, l2_projection_preserves_aligned_unit_jump) {
    auto mesh = Triangulation<2, 2>::UnitSquare(3);
    FeSpace space(mesh, DG<1, 1>);
    TrialFunction u(space);
    TestFunction v(space);
    auto step = [](const Eigen::Vector2d& point) { return point[0] < 0.5 ? 0.0 : 1.0; };
    ScalarField<2, decltype(step)> step_field(step);

    Eigen::SparseMatrix<double> mass = integral(mesh)(u * v).assemble();
    Eigen::VectorXd rhs = integral(mesh, QS2DP6)(step_field * v).assemble();
    mass.makeCompressed();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(mass);
    ASSERT_EQ(solver.info(), Eigen::Success);
    Eigen::VectorXd coefficients = solver.solve(rhs);
    ASSERT_EQ(solver.info(), Eigen::Success);

    // the aligned unit step belongs to DG1, so its L2 projection must retain the interface jump
    Eigen::SparseMatrix<double> jump_form = integral(mesh)(jump(u) * jump(v)).assemble();
    const double jump_squared_norm = coefficients.dot(jump_form * coefficients);
    EXPECT_LT((mass * coefficients - rhs).norm(), 1e-12);
    EXPECT_NEAR(jump_squared_norm, 1.0, 1e-12);
}

TEST(dg_poisson, smooth_manufactured_solution_has_expected_p1_rates) {
    constexpr double pi = std::numbers::pi;
    auto exact_solution = [](const Eigen::Matrix<double, 2, 1>& point) {
        return std::sin(pi * point[0]) * std::sin(pi * point[1]);
    };
    auto exact_gradient = [](const Eigen::Matrix<double, 2, 1>& point) {
        Eigen::Matrix<double, 2, 1> gradient;
        gradient << pi * std::cos(pi * point[0]) * std::sin(pi * point[1]),
          pi * std::sin(pi * point[0]) * std::cos(pi * point[1]);
        return gradient;
    };
    auto force = [](const Eigen::Matrix<double, 2, 1>& point) {
        return 2.0 * pi * pi * std::sin(pi * point[0]) * std::sin(pi * point[1]);
    };

    const dg_error coarse = solve_dg_poisson(5, force, exact_solution, exact_gradient);
    const dg_error medium = solve_dg_poisson(9, force, exact_solution, exact_gradient);
    const dg_error fine = solve_dg_poisson(17, force, exact_solution, exact_gradient);

    ASSERT_TRUE(std::isfinite(coarse.l2) && std::isfinite(medium.l2) && std::isfinite(fine.l2));
    ASSERT_TRUE(
      std::isfinite(coarse.broken_h1) && std::isfinite(medium.broken_h1) && std::isfinite(fine.broken_h1));
    EXPECT_EQ(fine.n_dofs, 1536);
    EXPECT_GT(convergence_rate(medium.l2, fine.l2), 1.7);
    EXPECT_GT(convergence_rate(medium.broken_h1, fine.broken_h1), 0.85);
    EXPECT_LT(fine.l2, medium.l2);
    EXPECT_LT(medium.l2, coarse.l2);
}

TEST(dg_poisson, solves_aligned_discontinuous_force_benchmark) {
    constexpr double pi = std::numbers::pi;
    constexpr double alpha = 1.0;
    auto g = [](double x) {
        const double base = x * (1.0 - x);
        if (x <= 0.5) return base;
        const double shifted = x - 0.5;
        return base + alpha * shifted * shifted * (1.0 - x);
    };
    auto g_prime = [](double x) {
        const double base = 1.0 - 2.0 * x;
        if (x <= 0.5) return base;
        const double shifted = x - 0.5;
        return base + alpha * (2.0 * shifted * (1.0 - x) - shifted * shifted);
    };
    auto exact_solution = [g](const Eigen::Matrix<double, 2, 1>& point) {
        return g(point[0]) * std::sin(pi * point[1]);
    };
    auto exact_gradient = [g, g_prime](const Eigen::Matrix<double, 2, 1>& point) {
        Eigen::Matrix<double, 2, 1> gradient;
        gradient << g_prime(point[0]) * std::sin(pi * point[1]),
          pi * g(point[0]) * std::cos(pi * point[1]);
        return gradient;
    };
    auto force = [g](const Eigen::Matrix<double, 2, 1>& point) {
        const double second_derivative_correction = point[0] <= 0.5 ? 0.0 : alpha * (4.0 - 6.0 * point[0]);
        return (pi * pi * g(point[0]) + 2.0 - second_derivative_correction) * std::sin(pi * point[1]);
    };

    Eigen::Vector2d interface_left(0.5, 0.5);
    Eigen::Vector2d interface_right(std::nextafter(0.5, 1.0), 0.5);
    EXPECT_NEAR(force(interface_right) - force(interface_left), -alpha, 1e-12);

    const dg_error coarse = solve_dg_poisson(5, force, exact_solution, exact_gradient);
    const dg_error medium = solve_dg_poisson(9, force, exact_solution, exact_gradient);
    const dg_error fine = solve_dg_poisson(17, force, exact_solution, exact_gradient);

    ASSERT_TRUE(std::isfinite(coarse.l2) && std::isfinite(medium.l2) && std::isfinite(fine.l2));
    ASSERT_TRUE(
      std::isfinite(coarse.broken_h1) && std::isfinite(medium.broken_h1) && std::isfinite(fine.broken_h1));
    EXPECT_GT(convergence_rate(medium.l2, fine.l2), 1.7);
    EXPECT_GT(convergence_rate(medium.broken_h1, fine.broken_h1), 0.85);
    EXPECT_LT(fine.l2, medium.l2);
    EXPECT_LT(medium.l2, coarse.l2);
}

}   // namespace fdapde
