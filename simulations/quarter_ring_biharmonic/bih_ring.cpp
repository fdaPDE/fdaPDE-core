#include "isogeometric.h"
#include "exact_solution.h"
#include "../helpers.h"

using namespace fdapde;

int main() {

    constexpr int M = 2;
    using Vec = Eigen::Matrix<double, M, 1>;
    using Fun = std::function<double(const Vec&)>;

    std::string folder = "results/";


    std::string save_path = "../" + folder  ;
    std::filesystem::create_directories(save_path);
    std::ofstream file(save_path + "L2_error.csv");
    file << "h_max,L2_error,H1_error,H2_error\n";

    std::vector<int> ref = {1, 2, 3, 4, 5, 6};

    auto f_exact = bih_ring::make_u_exact();
    auto u = bih_ring::make_rhs();
    auto df_exact = bih_ring::make_grad_u_exact();
    auto ddf_exact = bih_ring::make_hessian_u_exact();

    for (const auto& r : ref) {

        auto mesh = IsoMesh<2, 2>::quarter_ring();
        mesh.refine_knots({r, r});
        double h_max = mesh.h_max();

        IsoSpace Vh(mesh);
        TrialFunction f(Vh);
        TestFunction v(Vh);

        auto a = integral(mesh,QGL2DP9)(laplacian(f)*laplacian(v));
        auto F = integral(mesh, QGL2DP9)(u * v);

        auto& dof_handler = Vh.dof_handler();
        Eigen::SparseMatrix<double> A = a.assemble();
        auto b = F.assemble();

        dof_handler.set_clamped_hom_constraint();
        dof_handler.enforce_constraints(A,b);

        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        auto uh_full = solver.solve(b);

        IsoFunction solution(Vh);
        solution = uh_full;

        ScalarField<M> err_physical(
            [&](const Vec& p) {
                auto u = mesh.invert_point(p, 2);
                auto err = solution(u) - f_exact(p);
                return err * err;
            });


        ScalarField<M> err_H1physical(
            [&](const Vec& p) {
                auto u = mesh.invert_point(p, 2);
                Eigen::Vector2d grad_exact;
                grad_exact(0) = df_exact(p)(0,0);
                grad_exact(1) = df_exact(p)(1,0);

                auto err = solution.phys_grad(u) - grad_exact;
                return err.squaredNorm();
            });


        ScalarField<M> err_H2physical(
            [&](const Vec& p) {
                auto u = mesh.invert_point(p, 2);
                Eigen::Matrix2d hess_exact;
                hess_exact(0,0) = ddf_exact(p)(0,0);
                hess_exact(0,1) = ddf_exact(p)(0,1);
                hess_exact(1,0) = ddf_exact(p)(1,0);
                hess_exact(1,1) = ddf_exact(p)(1,1);
                auto diff_mat = solution.phys_hess(u) - hess_exact;
                return diff_mat(0, 0) * diff_mat(0, 0) + diff_mat(1, 1) * diff_mat(1, 1)
                                 + 2 * diff_mat(0, 1) * diff_mat(0, 1);
            });



        auto errorL2 = std::sqrt(integral(mesh, QGL2DP9)(err_physical));
        auto errorH1 = std::sqrt(errorL2*errorL2 + integral(mesh, QGL2DP9)( err_H1physical ));
        auto errorH2 =  std::sqrt(errorH1*errorH1 + integral(mesh, QGL2DP9)(err_H2physical));

        file << h_max << "," << errorL2 << "," << errorH1 <<","<<errorH2<< "\n";

        std::string level_path = save_path + "ref" + std::to_string(r) + "/mesh/";
        helpers::export_mesh(mesh, level_path);
        std::string solution_path = save_path + "ref" + std::to_string(r) + "/solution/";
        helpers::export_results(mesh, solution, solution_path, 10);

        std::cout << "\n===========================================\n";
        std::cout << "Refinement level: " << r << "\n";
        std::cout << "Number of cells : " << mesh.n_cells() << "\n";
        std::cout << "h_max           : " << h_max << "\n";
        std::cout << "L2 error        : " << errorL2 << "\n";
        std::cout << "H1 error        : " << errorH1 << "\n";
        std::cout << "H2 error        : " << errorH2 << "\n";
        std::cout << "Mesh exported to: " << level_path << "\n";
        std::cout << "PDE results to  : " << solution_path << "\n";
        std::cout << "===========================================\n";
    }

    return 0;
}