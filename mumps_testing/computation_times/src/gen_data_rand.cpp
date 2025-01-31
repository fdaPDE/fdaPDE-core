#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>

#include "../../../fdaPDE/linear_algebra/mumps.h"

using namespace std::chrono;
using namespace fdapde::mumps;
using namespace Eigen;

constexpr bool Seeded = true;
constexpr int Seed = 42;

SparseMatrix<double> generateFullRankMatrix(int size, double density, std::mt19937& gen);
SparseMatrix<double> generateSpdMatrix(int size, double density, std::mt19937& gen);
void printSparsityPattern(const SparseMatrix<double>& A, std::string filename);
void saveCSV(std::string dst_name);

int main(int argc, char* argv[]) {
    std::ofstream file;
    std::string filename = "../data/temp.csv";
    int Iterations = 1;
    std::vector<int> N = {10, 50, 100, 500, 1000, 5000};
    std::vector<double> dens = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005};

    std::vector<std::string> solvers = {"SparseLU", "MumpsLU", "MumpsBLR"};
    std::vector<std::string> spd_solvers = {"SimplicialLLT", "MumpsLDLT"};

    SparseMatrix<double> A;
    VectorXd x, b;

    std::mt19937 gen;
    if (Seeded) {
        gen.seed(Seed);
    } else {
        gen.seed(std::random_device()());
    }

    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix<SparseMatrix<double>, Dynamic, Dynamic> Matrices(N.size(), dens.size());
    for (size_t i = 0; i < N.size(); ++i) {
        for (size_t j = 0; j < dens.size(); ++j) {
            Matrices(i, j) = generateFullRankMatrix(N[i], dens[j], gen);
            if (rank == 0) {
                std::cout << "Generated matrix of size " << N[i] << "x" << N[i] << " with density " << dens[j] << "\n";
                if (!std::filesystem::exists("../data/sparsity_patterns/rand")) {
                    std::filesystem::create_directory("../data/sparsity_patterns/rand");
                }
                printSparsityPattern(
                  Matrices(i, j), "../data/sparsity_patterns/rand/rand_" + std::to_string(N[i]) + "_" +
                                    std::to_string(dens[j]) + ".csv");
            }
        }
    }

    Matrix<SparseMatrix<double>, Dynamic, Dynamic> MatricesSpd(N.size(), dens.size());
    for (size_t i = 0; i < N.size(); ++i) {
        for (size_t j = 0; j < dens.size(); ++j) {
            MatricesSpd(i, j) = generateSpdMatrix(N[i], dens[j], gen);
            if (rank == 0) {
                std::cout << "Generated SPD matrix of size " << N[i] << "x" << N[i] << " with density " << dens[j]
                          << "\n";
                if (!std::filesystem::exists("../data/sparsity_patterns/rand_spd")) {
                    std::filesystem::create_directory("../data/sparsity_patterns/rand_spd");
                }
                printSparsityPattern(
                  MatricesSpd(i, j), "../data/sparsity_patterns/rand_spd/rand_spd_" + std::to_string(N[i]) + "_" +
                                       std::to_string(dens[j]) + ".csv");
            }
        }
    }

    if (rank == 0) {
        file.open(filename);
        if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
        file << "Solver,N,density,time\n";
    }
    std::vector<std::string> temp = {"wrapped", "not-wrapped"};
    for (auto solver : temp) {
        for (size_t i = 0; i < N.size(); ++i) {
            for (size_t j = 0; j < dens.size(); ++j) {
                A = Matrices(i, j);
                b = VectorXd::Ones(N[i]);

                for (int n = 0; n < Iterations; ++n) {
                    if (solver == "wrapped") {
                        auto t1 = high_resolution_clock::now();
                        MumpsLU mumps(A);
                        x = mumps.solve(b);
                        auto t2 = high_resolution_clock::now();
                        duration<double> duration = t2 - t1;
                        MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                        if (rank == 0) {
                            std::cout << "Wrapped" << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
                            file << "Wrapped" << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
                        }
                    }
                    if (solver == "not-wrapped") {
                        std::vector<int> col_indices;
                        std::vector<int> row_indices;
                        std::vector<double> values;
                        col_indices.reserve(A.nonZeros());
                        row_indices.reserve(A.nonZeros());
                        values.reserve(A.nonZeros());
                        MatrixXd buff = b;
                        for (int k = 0; k < A.outerSize(); ++k) {
                            for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
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
                        MPI_Bcast(buff.data(), buff.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        duration<double> duration = t2 - t1;
                        MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                        if (rank == 0) {
                            std::cout << "Not-wrapped" << "," << N[i] << "," << dens[j] << "," << duration.count()
                                      << "\n";
                            file << "Not-wrapped" << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
                        }
                    }
                }
            }
        }
    }
    return 0;
    if (rank == 0) {
        file.close();
        saveCSV("../data/rand_computation_times_vs_raw.csv");
    }

    if (rank == 0) {
        file.open(filename);
        if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
        file << "Solver,N,density,time\n";
    }
    for (auto solver : solvers) {
        for (size_t i = 0; i < N.size(); ++i) {
            for (size_t j = 0; j < dens.size(); ++j) {
                A = Matrices(i, j);
                b = VectorXd::Ones(N[i]);

                for (int n = 0; n < Iterations; ++n) {
                    auto t1 = high_resolution_clock::now();
                    if (solver == "MumpsLU") {
                        MumpsLU mumps(A);
                        x = mumps.solve(b);
                    }
                    if (solver == "MumpsBLR") {
                        MumpsBLR mumps(A);
                        x = mumps.solve(b);
                    }
                    if (solver == "SparseLU") {
                        SparseLU<SparseMatrix<double>> solver(A);
                        x = solver.solve(b);
                    }
                    auto t2 = high_resolution_clock::now();
                    duration<double> duration = t2 - t1;
                    MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                    if (rank == 0) {
                        std::cout << solver << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
                        file << solver << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
                    }
                }
            }
        }
    }
    if (rank == 0) {
        file.close();
        saveCSV("../data/rand_computation_times.csv");
    }

    if (rank == 0) {
        file.open(filename);
        if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
        file << "Solver,N,density,time\n";
    }
    for (auto solver : spd_solvers) {
        for (size_t i = 0; i < N.size(); ++i) {
            for (size_t j = 5; j < 6 /*dens.size()*/; ++j) {
                A = MatricesSpd(i, j);
                b = VectorXd::Ones(N[i]);

                for (int n = 0; n < Iterations; ++n) {
                    auto t1 = high_resolution_clock::now();
                    if (solver == "MumpsLDLT") {
                        MumpsLDLT mumps(A);
                        // mumps.mumpsIcntl()[13] = 100;
                        x = mumps.solve(b);
                    }
                    if (solver == "SimplicialLLT") {
                        SimplicialLLT<SparseMatrix<double>> solver(A);
                        x = solver.solve(b);
                    }
                    auto t2 = high_resolution_clock::now();
                    duration<double> duration = t2 - t1;
                    MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                    if (rank == 0) {
                        std::cout << solver << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
                        file << solver << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
                    }
                }
            }
        }
    }
    if (rank == 0) {
        file.close();
        saveCSV("../data/rand_computation_times_spd.csv");
    }

    if (rank == 0) {
        file.open(filename);
        if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
        file << "Solver,N,density,schur_size,time\n";
    }
    std::string schur_solver = "MumpsSchur";
    for (size_t i = 0; i < N.size(); ++i) {
        std::vector<int> schur_size;
        schur_size.reserve(i + 1);
        schur_size.push_back(N[i] / 5);
        for (int k = 0; k <= i; ++k) { schur_size.push_back(N[i] / N[k]); }
        for (size_t j = 0; j < 1 /*dens.size()*/; ++j) {
            A = Matrices(i, j);
            b = VectorXd::Ones(N[i]);

            for (int k = 0; k < schur_size.size(); ++k) {
                for (int n = 0; n < Iterations; ++n) {
                    auto t1 = high_resolution_clock::now();
                    MumpsSchur mumps(A, schur_size[k], WorkingHost);
                    x = mumps.solve(b);
                    auto t2 = high_resolution_clock::now();
                    duration<double> duration = t2 - t1;
                    MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                    if (rank == 0) {
                        std::cout << schur_solver << "," << N[i] << "," << dens[j] << "," << schur_size[k] << ","
                                  << duration.count() << "\n";
                        file << schur_solver << "," << N[i] << "," << dens[j] << "," << schur_size[k] << ","
                             << duration.count() << "\n";
                    }
                }
            }
        }
    }
    if (rank == 0) {
        file.close();
        saveCSV("../data/rand_computation_times_schur.csv");
    }

    // delete temporary file
    if (rank == 0) std::filesystem::remove(filename);

    MPI_Finalize();
    return 0;
}

SparseMatrix<double> generateFullRankMatrix(int size, double density, std::mt19937& gen) {
    SparseMatrix<double> A(size, size);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::vector<Triplet<double>> tripletList;
    int nnz = static_cast<int>(density * size * size);   // Approximate non-zeros
    for (int k = 0; k < nnz; ++k) {
        int i = gen() % size;
        int j = gen() % size;
        double value = dis(gen);
        tripletList.push_back(Triplet<double>(i, j, value));
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // Ensure full rank by adding a small value to the diagonal
    for (int i = 0; i < size; ++i) { A.coeffRef(i, i) += 1e-5; }

    return A;
}

SparseMatrix<double> generateSpdMatrix(int size, double density, std::mt19937& gen) {
    SparseMatrix<double> A(size, size);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::vector<Triplet<double>> tripletList;
    int nnz = static_cast<int>(density * size * size / 2);   // Approximate non-zeros
    for (int k = 0; k < nnz; ++k) {
        int i = gen() % size;
        int j = gen() % size;
        double value = dis(gen);
        if (i <= j) {
            tripletList.push_back(Triplet<double>(i, j, value));
            if (i != j) {
                tripletList.push_back(Triplet<double>(j, i, value));   // Ensure symmetry
            }
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // Make it symmetric positive definite by adding a multiple of the identity (large enough to ensure positive
    // definiteness)
    for (int i = 0; i < size; ++i) { A.coeffRef(i, i) += size; }

    return A;
}

void printSparsityPattern(const SparseMatrix<double>& A, std::string filename) {
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

void saveCSV(std::string dst_name) {
    std::ifstream src("../data/temp.csv");
    std::ofstream dst(dst_name);
    dst << src.rdbuf();
    dst.close();
}
