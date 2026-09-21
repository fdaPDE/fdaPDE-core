// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>   // testing framework

using fdapde::core::NystromApproximation;
using fdapde::core::REVD;
using fdapde::core::RSVD;

using fdapde::core::RBKI;
using fdapde::core::RSI;

using fdapde::core::NysRBKI;
using fdapde::core::NysRSI;
using fdapde::core::RPChol;
#include "utils/utils.h"
using fdapde::testing::almost_equal;

// migrated to tests/integration/randomized_eigen.cpp: randomized_eigen.square_spectrum

// migrated to tests/integration/randomized_eigen.cpp: randomized_eigen.rectangular_spectrum

// migrated to tests/integration/randomized_eigen.cpp: randomized_eigen.full_rank_psd

// migrated to tests/integration/randomized_eigen.cpp: randomized_eigen.deficient_psd

TEST(nys_approximation, block_equal_one){
    DMatrix<double> A = DMatrix<double>::Random(40,20);
    A = A*A.transpose();
    int block_sz = 1;
    unsigned int seed = fdapde::random_seed; double tol = 1e-3;

    NystromApproximation<DMatrix<double>> rp_chol(std::make_unique<RPChol<DMatrix<double>>>(seed,tol));

    rp_chol.compute(A,block_sz);

    EXPECT_TRUE((A-rp_chol.factor()*rp_chol.factor().transpose()).norm() < tol*A.norm());
}

TEST(nys_approximation, block_larger_than_one){
    DMatrix<double> A = DMatrix<double>::Random(40,40);
    A = A*A.transpose();
    int block_sz = 7;
    unsigned int seed = fdapde::random_seed; double tol = 1e-3;

    NystromApproximation<DMatrix<double>> rp_chol(std::make_unique<RPChol<DMatrix<double>>>(seed,tol));
    rp_chol.compute(A,block_sz);

    EXPECT_TRUE((A-rp_chol.factor()*rp_chol.factor().transpose()).norm() < tol*A.norm());
}