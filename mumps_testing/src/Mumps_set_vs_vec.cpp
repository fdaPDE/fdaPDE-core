#include <gtest/gtest.h>

#include <unsupported/Eigen/SparseExtra>

#include "../../fdaPDE/linear_algebra.h"
#include "utils/rand_indices.h"

using namespace fdapde::mumps;
using namespace Eigen;

#include "../../test/src/utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;

#include <chrono>
using namespace std::chrono;

TEST(Mumps_set_vs_vector, inverse_elements) {
    int N = 100;
    int rank;
    high_resolution_clock::time_point t1, t2;
    microseconds time_span_set = microseconds(0);
    microseconds time_span_vector = microseconds(0);
    for (int i = 0; i < N; i++) {
        SparseMatrix<double> A;
        Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

        MumpsLU<SparseMatrix<double>> solver(A);
        if (i == 0) { rank = solver.getProcessRank(); }
        std::set<std::pair<int, int>> elements;
        randomInvIndices(elements, A.rows());
        t1 = high_resolution_clock::now();
        std::vector<Triplet<double>> inv_elements = solver.inverseElements(elements);
        t2 = high_resolution_clock::now();
        time_span_set += duration_cast<microseconds>(t2 - t1);
    }

    for (int i = 0; i < N; i++) {
        SparseMatrix<double> A;
        Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

        MumpsLU<SparseMatrix<double>> solver(A);
        if (i == 0) { rank = solver.getProcessRank(); }
        std::vector<std::pair<int, int>> elements;
        randomInvIndices(elements, A.rows());
        t1 = high_resolution_clock::now();
        std::vector<Triplet<double>> inv_elements = solver.inverseElements(elements);
        t2 = high_resolution_clock::now();
        time_span_vector += duration_cast<microseconds>(t2 - t1);
    }

    if (rank == 0) {
        std::cout << "Total Time for set: " << time_span_set.count() << " microseconds" << std::endl;
        std::cout << "Total Time for vector: " << time_span_vector.count() << " microseconds" << std::endl;
        if (time_span_set.count() < time_span_vector.count()) {
            std::cout << "Set is faster by " << time_span_vector.count() - time_span_set.count() << " microseconds"
                      << std::endl;
        } else {
            std::cout << "Vector is faster by " << time_span_set.count() - time_span_vector.count() << " microseconds "
                      << std::endl;
        }
    }
}

TEST(Mumps_set_vs_vector, schur) {
    int N = 100;
    int rank;
    high_resolution_clock::time_point t1, t2;
    microseconds time_span_set = microseconds(0);
    microseconds time_span_vector = microseconds(0);
    for (int i = 0; i < N; i++) {
        SparseMatrix<double> A(1000, 1000);
        MumpsSchur<SparseMatrix<double>> solver;
        if (i == 0) { rank = solver.getProcessRank(); }
        std::set<int> indices;
        randomSchurIndices(indices, A.rows());
        t1 = high_resolution_clock::now();
        solver.setSchurIndices(indices);
        t2 = high_resolution_clock::now();
        time_span_set += duration_cast<microseconds>(t2 - t1);
    }

    for (int i = 0; i < N; i++) {
        SparseMatrix<double> A(1000, 1000);
        MumpsSchur<SparseMatrix<double>> solver;
        if (i == 0) { rank = solver.getProcessRank(); }
        std::vector<int> indices;
        randomSchurIndices(indices, A.rows());
        t1 = high_resolution_clock::now();
        solver.setSchurIndices(indices);
        t2 = high_resolution_clock::now();
        time_span_vector += duration_cast<microseconds>(t2 - t1);
    }

    if (rank == 0) {
        std::cout << "Total Time for set: " << time_span_set.count() << " microseconds" << std::endl;
        std::cout << "Total Time for vector: " << time_span_vector.count() << " microseconds" << std::endl;
        if (time_span_set.count() < time_span_vector.count()) {
            std::cout << "Set is faster by " << time_span_vector.count() - time_span_set.count() << " microseconds"
                      << std::endl;
        } else {
            std::cout << "Vector is faster by " << time_span_set.count() - time_span_vector.count() << " microseconds"
                      << std::endl;
        }
    }
}
