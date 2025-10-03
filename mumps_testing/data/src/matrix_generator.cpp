#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <random>

using namespace Eigen;

SparseMatrix<double> generateSparseSPD(int size, double density, std::mt19937& gen) {
    SparseMatrix<double> A(size, size);

    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // Fill upper triangle of the matrix
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

SparseMatrix<double>
generateSparseFullRank(int size, double density, std::mt19937& gen, std::vector<Triplet<double>>& tripletList) {
    SparseMatrix<double> A(size, size);

    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // Fill the matrix with random values
    // std::vector<Triplet<double>> tripletList;
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

SparseMatrix<double> generateSparseRankDeficient(int size, double density, std::mt19937& gen) {
    SparseMatrix<double> A(size, size);

    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // Fill the matrix with random values
    std::vector<Triplet<double>> tripletList;
    int nnz = static_cast<int>(density * size * size);   // Approximate non-zeros
    for (int k = 0; k < nnz; ++k) {
        int i = gen() % size;
        int j = gen() % size;
        double value = dis(gen);
        tripletList.push_back(Triplet<double>(i, j, value));
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // set a random number of rows to be equal to the first one
    int num_rows = 2 + gen() % (size /2);
    for (int n = 0; n < num_rows; ++n) {
        for (int i = 0; i < size; ++i) {
            A.coeffRef(n, i) = 1;
        }
    }

    return A;
}

SparseMatrix<double>
generateFromPattern(int size, const std::vector<Triplet<double>>& tripletList, std::mt19937& gen, bool adjust = true) {
    SparseMatrix<double> B(size, size);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::vector<Triplet<double>> tripletListB;
    for (const auto& triplet : tripletList) {
        tripletListB.push_back(Triplet<double>(triplet.row(), triplet.col(), dis(gen)));
    }
    B.setFromTriplets(tripletListB.begin(), tripletListB.end());
    if (adjust) {
        for (int i = 0; i < size; ++i) { B.coeffRef(i, i) += 1e-5; }
    }
    return B;
}

void saveMatrixMarket(SparseMatrix<double>& A, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    A.makeCompressed();

    file << "%%MatrixMarket matrix coordinate real\n";
    file << A.rows() << " " << A.cols() << " " << A.nonZeros() << "\n";

    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            file << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value() << "\n";
        }
    }

    file.close();
}

int main(int argc, char* argv[]) {
    int size = 100;          // Matrix dimension
    double density = 0.1;   // Density of the sparse matrix

    std::mt19937 gen;
    if (argc > 1) {
        int seed = std::stoi(argv[1]);
        gen.seed(seed);
        std::cout << "Using provided seed: " << seed << std::endl;
    } else {
        std::random_device rd;
        gen.seed(rd());
        std::cout << "Using non-deterministic seed" << std::endl;
    }

    SparseMatrix<double> spdMatrix = generateSparseSPD(size, density, gen);
    SparseMatrix<double> rankDeficientMatrix = generateSparseRankDeficient(size, density, gen);

    saveMatrixMarket(spdMatrix, "../matrix_spd.mtx");
    saveMatrixMarket(rankDeficientMatrix, "../matrix_deficient.mtx");

    std::vector<Triplet<double>> tripletList;
    SparseMatrix<double> fullRankMatrix = generateSparseFullRank(size, density, gen, tripletList);
    SparseMatrix<double> fullRankMatrix2 = generateFromPattern(size, tripletList, gen);

    saveMatrixMarket(fullRankMatrix, "../matrix_fullrank.mtx");
    saveMatrixMarket(fullRankMatrix2, "../matrix_fullrank2.mtx");

    std::cout << "Matrices saved to matrix_spd.mtx, matrix_deficient.mtx, matrix_fullrank.mtx, matrix_fullrank2.mtx"
              << std::endl;

    // Calculate and save inverse of the symmetric positive definite matrix
    SimplicialLLT<SparseMatrix<double>> spdSolver(spdMatrix);
    if (spdSolver.info() == Success) {
        SparseMatrix<double> spdInverse = spdSolver.solve(MatrixXd::Identity(size, size)).sparseView();
        saveMatrixMarket(spdInverse, "../matrix_spd_inv.mtx");
        std::cout << "Inverse of SPD matrix saved to matrix_spd_inv.mtx" << std::endl;
    } else {
        std::cerr << "Failed to compute inverse of SPD matrix" << std::endl;
    }

    // Calculate and save inverse of the full-rank matrix
    SparseLU<SparseMatrix<double>> fullRankSolver(fullRankMatrix);
    if (fullRankSolver.info() == Success) {
        SparseMatrix<double> fullRankInverse = fullRankSolver.solve(MatrixXd::Identity(size, size)).sparseView();
        saveMatrixMarket(fullRankInverse, "../matrix_fullrank_inv.mtx");
        std::cout << "Inverse of full-rank matrix saved to matrix_fullrank_inv.mtx" << std::endl;
    } else {
        std::cerr << "Failed to compute inverse of full-rank matrix" << std::endl;
    }

    return 0;
}
