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

#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/linear_algebra.h>
using fdapde::core::RSVD;
using fdapde::core::REVD;
using fdapde::core::NystromApproximation;

using fdapde::core::RSI;
using fdapde::core::RBKI;

using fdapde::core::NysRSI;
using fdapde::core::NysRBKI;
using fdapde::core::RPChol;

#include "utils/utils.h"
using fdapde::testing::almost_equal;

TEST(rand_svd_test, square_test){
    DMatrix<double> A = DMatrix<double>::Random(20,20);
    int tr_rank = 3;
    unsigned int seed = fdapde::random_seed;
    double tol = 1e-3;

    RSVD<DMatrix<double>> rsi(std::make_unique<RSI<DMatrix<double>>>(seed,tol));
    RSVD<DMatrix<double>> rbki(std::make_unique<RBKI<DMatrix<double>>>(seed,tol));
    Eigen::JacobiSVD<DMatrix<double>> jacobi_svd;

    rsi.compute(A,tr_rank);
    rbki.compute(A,tr_rank);
    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rsi.singularValues()).template lpNorm<2>() < tol);
    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rbki.singularValues()).template lpNorm<2>() < tol);
}

TEST(rand_svd_test, rect_test){
    DMatrix<double> A = DMatrix<double>::Random(10,20);
    int tr_rank = 3;
    unsigned int seed = fdapde::random_seed;
    double tol = 1e-3;

    RSVD<DMatrix<double>> rsi(std::make_unique<RSI<DMatrix<double>>>(seed,tol));
    RSVD<DMatrix<double>> rbki(std::make_unique<RBKI<DMatrix<double>>>(seed,tol));
    Eigen::JacobiSVD<DMatrix<double>> jacobi_svd;

    rsi.compute(A,tr_rank);
    rbki.compute(A,tr_rank);
    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rsi.singularValues()).template lpNorm<2>() < tol);
    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rbki.singularValues()).template lpNorm<2>() < tol);

    rsi.compute(A.transpose(),tr_rank);
    rbki.compute(A.transpose(),tr_rank);
    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rsi.singularValues()).template lpNorm<2>() < tol);
    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-rbki.singularValues()).template lpNorm<2>() < tol);
}

TEST(rand_evd_test, full_rank){
    DMatrix<double> A = DMatrix<double>::Random(20,20);
    A = A*A.transpose();
    int tr_rank = 3;
    unsigned int seed = fdapde::random_seed; double tol = 1e-4;

    REVD<DMatrix<double>> nys_rsi(std::make_unique<NysRSI<DMatrix<double>>>(seed,tol));
    REVD<DMatrix<double>> nys_rbki(std::make_unique<NysRBKI<DMatrix<double>>>(seed,tol));
    Eigen::JacobiSVD<DMatrix<double>> jacobi_svd;

    nys_rsi.compute(A,tr_rank);
    nys_rbki.compute(A,tr_rank);
    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-nys_rsi.eigenValues()).template lpNorm<2>() < tol);
    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-nys_rbki.eigenValues()).template lpNorm<2>() < tol);
}

TEST(rand_evd_test, rank_deficient){
    DMatrix<double> A = DMatrix<double>::Random(40,20);
    A = A*A.transpose();
    int tr_rank = 3;
    unsigned int seed = fdapde::random_seed; double tol = 1e-4;

    REVD<DMatrix<double>> nys_rsi(std::make_unique<NysRSI<DMatrix<double>>>(seed,tol));
    REVD<DMatrix<double>> nys_rbki(std::make_unique<NysRBKI<DMatrix<double>>>(seed,tol));
    Eigen::JacobiSVD<DMatrix<double>> jacobi_svd;

    nys_rsi.compute(A,tr_rank);
    nys_rbki.compute(A,tr_rank);
    jacobi_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-nys_rsi.eigenValues()).template lpNorm<2>() < tol);
    EXPECT_TRUE((jacobi_svd.singularValues().head(tr_rank)-nys_rbki.eigenValues()).template lpNorm<2>() < tol);
}

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

