
#include "isogeometric.h"
#include "exact_solution.h"
#include "../helpers.h"

using namespace fdapde;

int main() {

    constexpr int M = 3;
    using Vec = Eigen::Matrix<double, M, 1>;
    using Fun = std::function<double(const Vec&)>;

    std::string folder = "results/";


    std::string save_path = "../" + folder  ;
    std::filesystem::create_directories(save_path);
    std::ofstream file(save_path + "L2_error.csv");
    file << "h_max,L2_error,H1_error,H2_error\n";

    std::vector<int> ref = {0, 1, 2, 3, 4, 5};

    auto f_exact = bih_sphere::make_u_exact();
    auto u = bih_sphere::make_rhs();
    auto df_exact = bih_sphere::make_grad_u_exact();
    auto ddf_exact = bih_sphere::make_hessian_u_exact();

    for (const auto& r : ref) {

        auto mesh = IsoMesh<2, 3>::sphere();
        if (r > 0) mesh.refine_knots({r, r});
        double h_max = mesh.h_max();

        // Set a periodic Spline basis

        std::array<std::vector<double>,2> open_uniform_knots;
        std::array<int,2> basis_dims;
        std::array<int,2> new_degree;
        for(int i = 0; i < 2; i++){
            new_degree[i] = mesh.degree()[i] ;
        }
        for(int i = 0; i < 2; i++){
            open_uniform_knots[i] = pad_knots(mesh.param_nodes()[i], new_degree[i]);
            basis_dims[i] = open_uniform_knots[i].size() - new_degree[i] - 1;
        }
        MdArray<double, full_dynamic_extent_t<2>> unitary_weights;
        unitary_weights.resize(basis_dims);
        std::fill(unitary_weights.begin(), unitary_weights.end(), 1.0); 
        auto basis_pde = NurbsBasis<2>(open_uniform_knots, unitary_weights, new_degree, mesh.is_periodic()); //bas
        //
        

        IsoSpace Vh(mesh, basis_pde);
        TrialFunction f(Vh);
        TestFunction v(Vh);

        auto a = integral(mesh, QGL2DP9)(laplacian(f) * laplacian(v));
        auto m = integral(mesh, QGL2DP9)(v);
        auto F = integral(mesh, QGL2DP9)(u * v);

        auto& dof_handler = Vh.dof_handler();
        Eigen::SparseMatrix<double> A = a.assemble();
        auto b = F.assemble();
        auto c = m.assemble();

                
        dof_handler.enforce_periodic_constraints(A,b);
        dof_handler.enforce_periodic_constraints(c);

        // Solve system with constraint (e.g., for unique solution on closed surface)
        int counter = b.size();
        Eigen::SparseMatrix<double> Zero(1, 1);
        SparseBlockMatrix<double, 2, 2> D(A, c.sparseView(), c.transpose().sparseView(), Zero);

        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(D);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(counter + 1);
        rhs.head(counter) = b;
        Eigen::VectorXd uh_reduced = solver.solve(rhs).head(counter);

        Eigen::VectorXd uh_full = dof_handler.expand_solution(uh_reduced);

        // Create IsoFunction
        IsoFunction solution(Vh);
        solution = uh_full;

        ScalarField<M> err_physical(
            [&](const Vec& p) {
                auto u = mesh.invert_point(p, 5);
                auto err = solution(u) - f_exact(p);
                return err * err;
            });

        ScalarField<M> err_H1physical(
            [&](const Vec& p) {
                auto u = mesh.invert_point(p, 5);
                Eigen::Vector3d grad_exact;
                for (int i = 0; i < M; ++i)
                    grad_exact(i) = df_exact(p)(i,0);
                auto diff_vec = solution.phys_grad(u) - grad_exact;
                return diff_vec(0) * diff_vec(0) + diff_vec(1) * diff_vec(1) + diff_vec(2) * diff_vec(2);
            });

        ScalarField<M> err_H2physical(
            [&](const Vec& p) {
                auto u = mesh.invert_point(p, 5);
                Eigen::Matrix3d hess_exact;
                for (int i = 0; i < M; ++i)
                    for (int j = 0; j < M; ++j)
                        hess_exact(i,j) = ddf_exact(p)(i,j);
                auto diff_mat = solution.phys_hess(u) - hess_exact;
                return diff_mat(0, 0) * diff_mat(0, 0) + diff_mat(1, 1) * diff_mat(1, 1) + diff_mat(2, 2) * diff_mat(2, 2)
                    + 2 * (diff_mat(0, 1) * diff_mat(0, 1) + diff_mat(0, 2) * diff_mat(0, 2) + diff_mat(1, 2) * diff_mat(1, 2));
            });


        auto errorL2 = std::sqrt(integral(mesh, QGL2DP9)(err_physical * err_physical));
        auto errorH1 = std::sqrt(errorL2 * errorL2 + integral(mesh, QGL2DP9)(err_H1physical));
        auto errorH2 = std::sqrt(integral(mesh, QGL2DP9)(err_H2physical) + errorH1 * errorH1) ;

        file << h_max << "," << errorL2 << "," << errorH1 << "," << errorH2 << "\n";

        // Export mesh and solution
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