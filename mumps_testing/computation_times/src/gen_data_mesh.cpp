#include <chrono>
#include <filesystem>
#include <fstream>

#include "../../../fdaPDE/fields.h"
#include "../../../fdaPDE/finite_elements.h"
#include "../../../fdaPDE/geoframe.h"
#include "../../../fdaPDE/geometry.h"
#include "../../../fdaPDE/linear_algebra.h"
#include "utils/mesh_generation.h"

using namespace Eigen;
using namespace fdapde;
using namespace fdapde::mumps;
using namespace std::chrono;

void printSparsityPattern(const Eigen::SparseMatrix<double>& A, std::string filename);
void saveCSV(std::string input_filename, std::string output_filename);

int main() {
    /**
     * BOOLEANS FOR THE COMPUION TIMES REQUESTED
     */
    bool print_sparsity_patterns = false;
    bool generate_times = true;
    bool generate_spd_times = true;
    bool generate_schur_times = false;
    bool generate_raw_times = true;

    /**
     * SET THE OUTPUT DIRECTORY
     */
    std::string output_directory = "../data";

    /**
     * SET THE TEMPORARY FILENAME
     *
     * Make sure that the name in not already used as it will be overwritten.
     */
    std::string temp_filename = "temp.csv";

    /**
     * SET THE OUTPUT FILENAMES
     */
    std::string output_filename = "computation_times.csv";
    std::string output_filename_spd = "computation_times_spd.csv";
    std::string output_filename_schur = "computation_times_schur.csv";
    std::string output_filename_raw = "computation_times_raw.csv";

    /**
     * SET THE PROBLEM SIZE
     */
    std::vector<int> N = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    /**
     * SET THE SOLVERS TO BE USED
     *
     * Available solvers:
     * - SparseLU
     * - MumpsLU
     * - MumpsBLR
     *
     * Available symmetric solvers:
     * - SimplicialLLT
     * - MumpsLDLT
     */
    std::vector<std::string> solvers = {"SparseLU", "MumpsLU", "MumpsBLR"};
    std::vector<std::string> solvers_spd = {"SimplicialLLT", "MumpsLDLT"};

    /**
     * SET THE NUMBER OF ITERATIONS
     */
    int n_iter = 10;

    /**
     * SET THE PROBLEMS TO BE SOLVED
     *
     * Available problems:
     * - mass (SPD)
     * - laplacian (SPD)
     * - diffusion_transport (non-symmetric)
     */
    std::vector<std::string> problems = {"mass", "laplacian", "diffusion_transport"};
    std::vector<std::string> problems_spd = {"mass"};
    std::vector<std::string> problems_schur = {"mass"};

    // data structures
    std::vector<Eigen::SparseMatrix<double>> mass_matrices;
    std::vector<Eigen::SparseMatrix<double>> laplacian_matrices;
    std::vector<Eigen::SparseMatrix<double>> diffusion_transport_matrices;
    std::vector<Eigen::VectorXd> forces;

    // MPI initialization + rank (for output)
    MPI_Init(NULL, NULL);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) std::cout << "MPI initialized with " << size << " processes." << std::endl;

    // create output directory
    if (rank == 0) {
        if (!std::filesystem::exists(output_directory)) { std::filesystem::create_directory(output_directory); }
    }

    // fill data structures
    for (int n : N) {
        /**
         * Available mesh generators:
         * - meshInterval(a, b, n_nodes)
         * - meshUnitInterval(n_nodes)
         * - meshRectangle(a_x, b_x, a_y, b_y, n_nodes_x, n_nodes_y)
         * - meshSquare(a, b, n_nodes)
         * - meshUnitSquare(n_nodes)
         * - meshParallelepipiped(a_x, b_x, a_y, b_y, a_z, b_z, n_nodes_x, n_nodes_y, n_nodes_z)
         * - meshCube(a, b, n_nodes)
         * - meshUnitCube(n_nodes)
         */

        // Eigen::Eigen::VectorXd force(n*n);
        if (rank == 0) {
            auto mesh = meshUnitSquare(n);
            if (rank == 0) std::cout << "Generated mesh with " << n << " nodes" << std::endl;

            Triangulation</* tangent_space = */ 2, /* embedding_space = */ 2> unit_square(
              mesh.nodes, mesh.elements, mesh.boundary);

            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction v(Vh);
            auto a1 = integral(unit_square)(u * v);                   // mass matrix: SPD
            auto a2 = integral(unit_square)(dot(grad(u), grad(v)));   // laplacian discretization (Poisson): SPD

            Matrix<double, 2, 1> b(0.2, 0.2);

            auto a3 = integral(unit_square)(
              dot(grad(u), grad(v)) + dot(b, grad(u)) * v);   // diffusion-transport: non simmetrica

            mass_matrices.push_back(Eigen::SparseMatrix<double>(a1.assemble()));
            laplacian_matrices.push_back(Eigen::SparseMatrix<double>(a2.assemble()));
            diffusion_transport_matrices.push_back(Eigen::SparseMatrix<double>(a3.assemble()));
            if (rank == 0) std::cout << "Generated matrices for " << n << " nodes" << std::endl;

            // forcing term
            ScalarField<2, decltype([]([[maybe_unused]] const Vector2d& p) { return 1; })> f;
            auto F = integral(unit_square)(f * v);

            // force = F.assemble();
            forces.push_back(F.assemble());

            // linear system: R1 * u = f
        } else {
            mass_matrices.push_back(Eigen::SparseMatrix<double>());
            laplacian_matrices.push_back(Eigen::SparseMatrix<double>());
            diffusion_transport_matrices.push_back(Eigen::SparseMatrix<double>());

            forces.push_back(Eigen::VectorXd(n * n));
        }
        // MPI_Bcast(force.data(), n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // forces.push_back(force);
        if (rank == 0) std::cout << "Generated force for " << n << " nodes" << std::endl;
    }

    // print sparsity patterns of the matrices
    if (print_sparsity_patterns) {
        if (rank == 0) {
            std::cout << "Printing sparsity patterns" << std::endl;
            std::string sp_patterns_directory = output_directory + "/sparsity_patterns";
            if (!std::filesystem::exists(sp_patterns_directory)) {
                std::filesystem::create_directory(sp_patterns_directory);
            }
            if (!std::filesystem::exists(sp_patterns_directory + "/mass")) {
                std::filesystem::create_directory(sp_patterns_directory + "/mass");
            }
            if (!std::filesystem::exists(sp_patterns_directory + "/laplacian")) {
                std::filesystem::create_directory(sp_patterns_directory + "/laplacian");
            }
            if (!std::filesystem::exists(sp_patterns_directory + "/diffusion_transport")) {
                std::filesystem::create_directory(sp_patterns_directory + "/diffusion_transport");
            }
            for (int i = 0; i < N.size(); ++i) {
                printSparsityPattern(
                  mass_matrices[i], sp_patterns_directory + "/mass/mass_" + std::to_string(N[i]) + ".csv");
                printSparsityPattern(
                  laplacian_matrices[i],
                  sp_patterns_directory + "/laplacian/laplacian_" + std::to_string(N[i]) + ".csv");
                printSparsityPattern(
                  diffusion_transport_matrices[i],
                  sp_patterns_directory + "/diffusion_transport/diffusion_transport_" + std::to_string(N[i]) + ".csv");
            }
        }
    }

    // file manager for output
    std::ofstream file;
    std::string temp_path = output_directory + "/" + temp_filename;
    std::string output_path = output_directory + "/" + output_filename;
    std::string output_path_spd = output_directory + "/" + output_filename_spd;
    std::string output_path_schur = output_directory + "/" + output_filename_schur;
    std::string output_path_raw = output_directory + "/" + output_filename_raw;

    if (generate_times) {
        if (rank == 0) {
            std::cout << "Solving problems" << std::endl;
            file.open(temp_path);
            if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
            file << "N,problem,solver,time\n";
        }
        for (size_t i = 0; i < N.size(); ++i) {
            Eigen::SparseMatrix<double> A;
            for (const auto& problem : problems) {
                if (problem == "mass") { A = mass_matrices[i]; }
                if (problem == "laplacian") { A = laplacian_matrices[i]; }
                if (problem == "diffusion_transport") { A = diffusion_transport_matrices[i]; }
                for (const auto& solver : solvers) {
		    bool one_time_flag = false;
                    for (int iter = 0; iter < n_iter; ++iter) {
                        auto t1 = high_resolution_clock::now();
                        if (solver == "SparseLU" && rank == 0 && (N[i]<2500 || one_time_flag)) {
                            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(A);
                            Eigen::VectorXd x = solver.solve(forces[i]);
                        }
                        if (solver == "MumpsLU") {
                            MumpsLU solver(A);
                            Eigen::VectorXd x = solver.solve(forces[i]);
                        }
                        if (solver == "MumpsBLR") {
                            MumpsBLR solver(A);
                            Eigen::VectorXd x = solver.solve(forces[i]);
                        }
                        auto t2 = high_resolution_clock::now();
                        duration<double> duration = t2 - t1;
                        MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                        if (rank == 0 && !(solver == "SparseLU" && (N[i]>=2500 && !one_time_flag))) {
                            std::cout << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                            file << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                        }
			if (N[i]>=2500 && one_time_flag) one_time_flag = false;
                    }
                }
            }
        }
        if (rank == 0) {
            file.close();
            saveCSV(temp_path, output_path);
        }
    }

    if (generate_spd_times) {
        if (rank == 0) {
            std::cout << "Solving SPD problems" << std::endl;
            file.open(temp_path);
            if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
            file << "N,problem,solver,time\n";
        }
        for (size_t i = 0; i < N.size(); ++i) {
            Eigen::SparseMatrix<double> A;
            for (const auto& problem : problems_spd) {
                if (problem == "mass") { A = mass_matrices[i]; }
                if (problem == "laplacian") { A = laplacian_matrices[i]; }
                for (const auto& solver : solvers_spd) {
		    bool one_time_flag = true;
                    for (int iter = 0; iter < n_iter; ++iter) {
                        auto t1 = high_resolution_clock::now();
                        if (solver == "SimplicialLLT" && rank == 0 && (N[i]<2500 || one_time_flag)) {
                            Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(A);
                            Eigen::VectorXd x = solver.solve(forces[i]);
                        }
                        if (solver == "MumpsLDLT") {
                            MumpsLDLT solver(A);
                            Eigen::VectorXd x = solver.solve(forces[i]);
                        }
                        auto t2 = high_resolution_clock::now();
                        duration<double> duration = t2 - t1;
                        MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                        if (rank == 0 && !(solver == "SimplicialLLT" && (N[i]>=2500 && !one_time_flag))) {
                            std::cout << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                            file << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                        }
			if (N[i] >= 2500 && one_time_flag) one_time_flag = false;
                    }
                }
            }
        }
        if (rank == 0) {
            file.close();
            saveCSV(temp_path, output_path_spd);
        }
    }

    if (generate_schur_times) {
        if (rank == 0) {
            std::cout << "Solving Schur problems" << std::endl;
            file.open(temp_path);
            if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
            file << "N,problem,schur_size,time\n";
        }

        for (size_t i = 0; i < N.size(); ++i) {
            Eigen::SparseMatrix<double> A;
            for (const auto& problem : problems_schur) {
                if (problem == "mass") { A = mass_matrices[i]; }
                if (problem == "laplacian") { A = laplacian_matrices[i]; }
                if (problem == "diffusion_transport") { A = diffusion_transport_matrices[i]; }
                for (int schur_size = N[i]; schur_size < std::min(N[i]*10, N[i]*N[i]/2); schur_size += N[i]) {   // change step ???
                    for (int iter = 0; iter < n_iter; ++iter) {
                        auto t1 = high_resolution_clock::now();
                        MumpsSchur solver(A, schur_size);
                        Eigen::VectorXd x = solver.solve(forces[i]);
                        auto t2 = high_resolution_clock::now();
                        duration<double> duration = t2 - t1;
                        MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                        if (rank == 0) {
                            std::cout << N[i] << "," << problem << "," << schur_size << "," << duration.count() << "\n";
                            file << N[i] << "," << problem << "," << schur_size << "," << duration.count() << "\n";
                        }
                    }
                }
            }
        }
        if (rank == 0) {
            file.close();
            saveCSV(temp_path, output_path_schur);
        }
    }

    if (generate_raw_times) {
        if (rank == 0) {
            std::cout << "Solving with raw MUMPS" << std::endl;
            file.open(temp_path);
            if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
            file << "N,problem,solver,time\n";
        }

        std::vector<std::string> temp_solvers = {"Wrapped", "Non-wrapped"};
        for (size_t i = 0; i < N.size(); ++i) {
            Eigen::SparseMatrix<double> A;
            for (const auto& problem : problems) {
                if (problem == "mass") { A = mass_matrices[i]; }
                if (problem == "laplacian") { A = laplacian_matrices[i]; }
                if (problem == "diffusion_transport") { A = diffusion_transport_matrices[i]; }
                for (const auto& solver : temp_solvers) {
                    for (int iter = 0; iter < n_iter; ++iter) {
                        duration<double> duration;
                        if (solver == "Wrapped") {
                            auto t1 = high_resolution_clock::now();
                            MumpsLU solver(A);
                            Eigen::VectorXd x = solver.solve(forces[i]);
                            auto t2 = high_resolution_clock::now();
                            duration = t2 - t1;
                        }
                        if (solver == "Non-wrapped") {
                            std::vector<int> col_indices;
                            std::vector<int> row_indices;
                            std::vector<double> values;
                            col_indices.reserve(A.nonZeros());
                            row_indices.reserve(A.nonZeros());
                            values.reserve(A.nonZeros());
                            MatrixXd buff = forces[i];
                            for (int k = 0; k < A.outerSize(); ++k) {
                                for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                                    row_indices.push_back(it.row() + 1);
                                    col_indices.push_back(it.col() + 1);
                                    values.push_back(it.value());
                                }
                            }
                            auto t1 = high_resolution_clock::now();
                            DMUMPS_STRUC_C id;
                            id.comm_fortran = (MUMPS_INT)MPI_Comm_c2f(MPI_COMM_WORLD);
                            id.par = 0;
                            id.sym = 0;
                            id.job = -1;
                            dmumps_c(&id);
                            if (rank == 0) {
                                id.n = A.rows();
                                id.nnz = A.nonZeros();
                                id.irn = row_indices.data();
                                id.jcn = col_indices.data();
                                id.a = values.data();

                                id.nrhs = buff.cols();
                                id.lrhs = buff.rows();
                                id.rhs = const_cast<double*>(buff.data());
                            }
                            id.icntl[32] = 1;
                            id.icntl[0] = -1;
                            id.icntl[1] = -1;
                            id.icntl[2] = -1;
                            id.icntl[3] = 0;
                            id.job = 6;
                            dmumps_c(&id);
                            auto t2 = high_resolution_clock::now();
                            id.job = -2;
                            dmumps_c(&id);
                            duration = t2 - t1;
                        }
                        MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                        if (rank == 0) {
                            std::cout << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                            file << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                        }
                    }
                }
            }
        }
        if (rank == 0) {
            file.close();
            saveCSV(temp_path, output_path_raw);
        }
    }

    // delete temporary file
    if (rank == 0) {
        if (std::filesystem::exists(temp_path)) std::filesystem::remove(temp_path);
    }

    MPI_Finalize();
    return 0;
}

void printSparsityPattern(const Eigen::SparseMatrix<double>& A, std::string filename) {
    std::ofstream file(filename);
    file << "i,j\n";
    if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            if (A.coeff(i, j) != 0) { file << i << "," << j << "\n"; }
        }
    }
    file.close();
}

void saveCSV(std::string input_filename, std::string output_filename) {
    std::ifstream input_file(input_filename);
    std::ofstream output_file(output_filename);
    if (!input_file.is_open()) { throw std::runtime_error("Unable to open file for reading."); }
    if (!output_file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
    output_file << input_file.rdbuf();
    output_file.close();
}
