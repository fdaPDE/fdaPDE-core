#include <gtest/gtest.h>
#include <unsupported/Eigen/SparseExtra>
#include "isogeometric.h"
#include "utils/utils.h"

using namespace fdapde;
using SpMatrix = Eigen::SparseMatrix<double>;
using Vector = Eigen::VectorXd;

TEST(SplineTest, OpenBSplineBasis) {
    SpMatrix expected_basis, expected_deriv;
    Eigen::loadMarket(expected_basis, "../data/sp_basis_nonperiodic.mtx");
    Eigen::loadMarket(expected_deriv, "../data/sp_basis_first_der_nonperiodic.mtx");

    std::vector<double> knots = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0};
    int degree = 2;
    BSplineBasis basis(knots, degree, false);

    for (int j = 0; j < expected_basis.cols(); ++j) {
        double x = expected_basis.coeff(0, j);
        auto values = basis.evaluate_basis(x);
        auto ders = basis.evaluate_der_basis(x, 1);

        for (int i = 0; i < values.size(); ++i) {
            EXPECT_TRUE(isotesting::almost_equal(values[i], expected_basis.coeff(i + 1, j)));
            EXPECT_TRUE(isotesting::almost_equal(ders[1][i], expected_deriv.coeff(i + 1, j)));
        }
    }
}

TEST(SplineTest, PeriodicBSplineBasis) {
    SpMatrix expected;
    Eigen::loadMarket(expected, "../data/sp_basis_periodic.mtx");

    std::vector<double> knots = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0};
    int degree = 2;
    BSplineBasis basis(knots, degree, /* periodicity = */true);

    for (int j = 0; j < expected.cols(); ++j) {
        double x = expected.coeff(0, j);
        auto values = basis.evaluate_basis(x);

        for (int i = 0; i < values.size(); ++i) {
            EXPECT_TRUE(isotesting::almost_equal(values[i], expected.coeff(i + 1, j)));
        }
    }
}

TEST(NurbsTest, NurbsSBasis2D) {
    SpMatrix expected_eval, expected_grad, expected_hess;
    Eigen::loadMarket(expected_eval, "../data/nurbs2d_eval.mtx");
    Eigen::loadMarket(expected_grad, "../data/nurbs2d_grad.mtx");
    Eigen::loadMarket(expected_hess, "../data/nurbs2d_hess.mtx");

    std::array<std::vector<double>, 2> knots = {{
        {0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0},
        {0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0}
    }};
    std::array<int, 2> degree = {2, 2};
    MdArray<double, full_dynamic_extent_t<2>> weights(4, 4);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            weights(i, j) = 1.0 + 0.1 * i * j;
        }
    }

    NurbsBasis<2> basis(knots, weights, degree);

    for (int j = 0; j < expected_eval.cols(); ++j) {
        Eigen::Vector2d p(expected_eval.coeff(0, j), expected_eval.coeff(1, j));

        for (int i = 0; i < basis.size(); ++i) {
            EXPECT_TRUE(isotesting::almost_equal(basis[i](p), expected_eval.coeff(i + 2, j)));

            auto grad = basis[i].gradient(p);
            EXPECT_TRUE(isotesting::almost_equal(grad(0), expected_grad.coeff(2 + 2 * i, j)));
            EXPECT_TRUE(isotesting::almost_equal(grad(1), expected_grad.coeff(2 + 2 * i + 1, j)));

            auto hess = basis[i].hessian(p);
            EXPECT_TRUE(isotesting::almost_equal(hess(0, 0), expected_hess.coeff(2 + 4 * i + 0, j)));
            EXPECT_TRUE(isotesting::almost_equal(hess(0, 1), expected_hess.coeff(2 + 4 * i + 1, j)));
            EXPECT_TRUE(isotesting::almost_equal(hess(1, 0), expected_hess.coeff(2 + 4 * i + 2, j)));
            EXPECT_TRUE(isotesting::almost_equal(hess(1, 1), expected_hess.coeff(2 + 4 * i + 3, j)));
        }
    }
}

TEST(IsoMeshTest, GeometryEval) {
    SpMatrix expected;
    Eigen::loadMarket(expected, "../data/torus_eval_param.mtx");

    auto mesh = IsoMesh<2, 3>::torus();

    const int grid = 20;

    for (int j = 0; j < expected.cols(); ++j) {
        Eigen::Vector2d u;
        int u_i = j / (grid + 1);
        int v_i = j % (grid + 1);

        u(0) = static_cast<double>(u_i) / grid;
        u(1) = static_cast<double>(v_i) / grid;

        Eigen::Vector3d x = mesh.eval_param(u);

        EXPECT_TRUE(isotesting::almost_equal(x(0), expected.coeff(1, j)));
        EXPECT_TRUE(isotesting::almost_equal(x(1), expected.coeff(2, j)));
        EXPECT_TRUE(isotesting::almost_equal(x(2), expected.coeff(3, j)));
    }
}

TEST(IsoMeshTest, JacobianEval) {
    SpMatrix expected;
    Eigen::loadMarket(expected, "../data/torus_eval_param_jacobian.mtx");

    auto mesh = IsoMesh<2, 3>::torus();

    const int grid = 20;

    for (int j = 0; j < expected.cols(); ++j) {
        Eigen::Vector2d u;
        int u_i = j / (grid + 1);
        int v_i = j % (grid + 1);

        u(0) = static_cast<double>(u_i) / grid;
        u(1) = static_cast<double>(v_i) / grid;

        auto derivs = mesh.eval_param_derivatives(u);

        for (int d = 0; d < 3; ++d) {
            for (int i = 0; i < 2; ++i) {
                int row = 1 + d * 2 + i;
                EXPECT_TRUE(isotesting::almost_equal(derivs.first_derivative(d, i), expected.coeff(row, j)));
            }
        }
    }
}

TEST(IsoMeshTest, HessianEval) {
    SpMatrix expected;
    Eigen::loadMarket(expected, "../data/torus_eval_param_hessian.mtx");

    auto mesh = IsoMesh<2, 3>::torus();

    const int grid = 20;

    for (int j = 0; j < expected.cols(); ++j) {
        Eigen::Vector2d u;
        int u_i = j / (grid + 1);
        int v_i = j % (grid + 1);

        u(0) = static_cast<double>(u_i) / grid;
        u(1) = static_cast<double>(v_i) / grid;

        auto derivs = mesh.eval_param_derivatives(u, true);
        const auto& H = *(derivs.second_derivative);

        for (int d = 0; d < 3; ++d) {
            for (int i = 0; i < 2; ++i) {
                for (int k = 0; k < 2; ++k) {
                    int row = 1 + d * 4 + 2 * i + k;
                    EXPECT_TRUE(isotesting::almost_equal(H(d, i, k), expected.coeff(row, j)));
                }
            }
        }
    }
}

TEST(IsoMeshTest, InvertPoint) {
    SpMatrix input, output;
    Eigen::loadMarket(input, "../data/torus_invert_point_input.mtx");
    Eigen::loadMarket(output, "../data/torus_invert_point_output.mtx");

    auto mesh = IsoMesh<2, 3>::torus();

    for (int j = 0; j < input.cols(); ++j) {
        Eigen::Vector3d p;
        p << input.coeff(1, j), input.coeff(2, j), input.coeff(3, j);

        auto u = mesh.invert_point(p);

        EXPECT_TRUE(isotesting::almost_equal(u(0), output.coeff(1, j)));
        EXPECT_TRUE(isotesting::almost_equal(u(1), output.coeff(2, j)));
    }
}

TEST(IsoMeshTest, hRefinement) {
    auto mesh = IsoMesh<2, 3>::torus();

    const int grid = 10;
    std::vector<Eigen::Vector2d> u_points;

    for (int i = 0; i <= grid; ++i) {
        for (int j = 0; j <= grid; ++j) {
            u_points.emplace_back(i / double(grid), j / double(grid));
        }
    }

    std::vector<Eigen::Vector3d> original_points;
    for (const auto& u : u_points) {
        original_points.push_back(mesh.eval_param(u));
    }

    mesh.refine_knots({{1, 1}});  // Apply refinement

    for (size_t i = 0; i < u_points.size(); ++i) {
        Eigen::Vector3d refined = mesh.eval_param(u_points[i]);

        EXPECT_TRUE(isotesting::almost_equal(refined(0), original_points[i](0)));
        EXPECT_TRUE(isotesting::almost_equal(refined(1), original_points[i](1)));
        EXPECT_TRUE(isotesting::almost_equal(refined(2), original_points[i](2)));
    }
}

TEST(IsoMeshTest, IntegralUnit) {
    auto mesh = IsoMesh<2, 3>::sphere();
    mesh.refine_knots({{3, 3}});  // Apply refinement

    ScalarField<3, decltype([](const Eigen::Matrix<double, 3, 1>& p) { return 1; })> u;
    double result = integral(mesh, QGL2DP9)(u);

    EXPECT_TRUE(isotesting::almost_equal(result, 4 * M_PI, 1e-6));
}

TEST(PDETest, StiffnessMatrix) {
    auto mesh = IsoMesh<2, 2>::quarter_ring();

    mesh.refine_knots({{2, 2}});  // Apply refinement

    IsoSpace Vh(mesh);
    TrialFunction f(Vh);
    TestFunction v(Vh);

    auto a = integral(mesh, QGL2DP9)(dot(grad(f), grad(v)));
    SpMatrix A = a.assemble();

    SpMatrix A_correct;
    Eigen::loadMarket(A_correct, "../data/A_geopdes.mtx");

    EXPECT_TRUE(isotesting::almost_equal(A, A_correct));
}

TEST(PDETest, MassMatrix) {
    auto mesh = IsoMesh<2, 2>::quarter_ring();

    mesh.refine_knots({{2, 2}});  // Apply refinement

    IsoSpace Vh(mesh);
    TrialFunction f(Vh);
    TestFunction v(Vh);

    auto m = integral(mesh, QGL2DP9)(f*v);
    SpMatrix M = m.assemble();

    SpMatrix M_correct;
    Eigen::loadMarket(M_correct, "../data/M_geopdes.mtx");

    EXPECT_TRUE(isotesting::almost_equal(M, M_correct));
}

TEST(PDETest, RHSAssembly) {
    auto mesh = IsoMesh<2, 2>::quarter_ring();

    mesh.refine_knots({{2, 2}});  // Apply refinement

    IsoSpace Vh(mesh);
    TrialFunction f(Vh);
    TestFunction v(Vh);

    ScalarField<2, decltype([](const Eigen::Vector2d& p) {
        double x = p(0);
        double y = p(1);
        return 8 * x * y * (8 * x * x + 8 * y * y - 15);
    })> u;

    auto b = integral(mesh, QGL2DP9)(u * v);
    auto rhs = b.assemble();

    SpMatrix b_correct;
    Eigen::loadMarket(b_correct, "../data/b_geopdes.mtx");
    auto b_correct_full = b_correct.toDense();

    EXPECT_TRUE(isotesting::almost_equal(rhs, b_correct_full));
}

TEST(PDETest, BiharmonicMatrix) {
    auto mesh = IsoMesh<2, 2>::quarter_ring();
    mesh.refine_knots({{2, 2}});  // Apply refinement

    IsoSpace Vh(mesh);
    TrialFunction f(Vh);
    TestFunction v(Vh);

    auto a = integral(mesh, QGL2DP9)(laplacian(f) * laplacian(v));
    SpMatrix A = a.assemble();

    SpMatrix A_correct;
    Eigen::loadMarket(A_correct, "../data/A_bih_geopdes.mtx");

    EXPECT_TRUE(isotesting::almost_equal(A, A_correct));
}
