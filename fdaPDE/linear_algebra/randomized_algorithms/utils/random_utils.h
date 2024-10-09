//
// Created by Marco Galliani on 05/09/24.
//

#ifndef TEST_RSVD_RANDOM_UTILS_H
#define TEST_RSVD_RANDOM_UTILS_H

#include <random>

namespace fdapde{
namespace core{

//IterationPolicy
enum IterationPolicy{
    SubspaceIterations,
    ExtendedSubspaceIterations,
    BlockKrylovIterations,
    ExtendedBlockKrylovIterations,
    RandomlyPivotedCholesky};

//StoppingPolicy
enum StoppingPolicy{ReconstructionAccuracy, SingularValuesAccuracy};

// Definition of the gaussian matrix generator
inline DMatrix<double> GaussianMatrix(size_t rows, size_t cols, unsigned int seed=42, double sigma = 1.0){
    std::mt19937 rand_eng{seed};
    std::normal_distribution norm_distr{0.0,sigma};

    DMatrix<double> Values(rows,cols);
    //filling the vector with random values
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; j++){
            Values(i,j) = norm_distr(rand_eng);
        }
    }
    return Values;
}

}//core
}//fdpade


#endif //TEST_RSVD_RANDOM_UTILS_H
