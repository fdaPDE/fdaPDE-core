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
using fdapde::core::RandomizedSVD;
using fdapde::core::RandomizedRangeFinder;

using fdapde::core::RandomizedEVD;

using fdapde::core::IterationPolicy;
using fdapde::core::StoppingPolicy;

#include "utils/utils.h"
using fdapde::testing::almost_equal;

TEST(rand_svd_test, square_test){
    DMatrix<double> A = DMatrix<double>::Random(20,20);
    int tr_rank = 3;
    double tol = 1e-3; //default tolerance for RandomizedSVD

    RandomizedSVD<decltype(A),IterationPolicy::SubspaceIterations> rsi(A,tr_rank);
    RandomizedSVD<decltype(A),IterationPolicy::ExtendedSubspaceIterations> rsi_ext(A,tr_rank);
    RandomizedSVD<decltype(A),IterationPolicy::BlockKrylovIterations> rbki(A,tr_rank);
    RandomizedSVD<decltype(A),IterationPolicy::ExtendedBlockKrylovIterations> rbki_ext(A,tr_rank);
    Eigen::JacobiSVD<DMatrix<double>> svd(A);

    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi_ext.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki_ext.singularValues()).lpNorm<2>() < tol);

    //using a different constructor
    rsi = RandomizedSVD<decltype(A),IterationPolicy::SubspaceIterations>(A,tr_rank,2*tr_rank);
    rsi_ext = RandomizedSVD<decltype(A),IterationPolicy::ExtendedSubspaceIterations>(A,tr_rank,2*tr_rank);
    rbki = RandomizedSVD<decltype(A),IterationPolicy::BlockKrylovIterations>(A,tr_rank,1);
    rbki_ext = RandomizedSVD<decltype(A),IterationPolicy::ExtendedBlockKrylovIterations>(A,tr_rank,1);

    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi_ext.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki_ext.singularValues()).lpNorm<2>() < tol);

    //another constructor
    rsi = RandomizedSVD<decltype(A),IterationPolicy::SubspaceIterations>(tr_rank,tol);
    rsi_ext = RandomizedSVD<decltype(A),IterationPolicy::ExtendedSubspaceIterations>(tr_rank,tol);
    rbki = RandomizedSVD<decltype(A),IterationPolicy::BlockKrylovIterations>(tr_rank,tol);
    rbki_ext = RandomizedSVD<decltype(A),IterationPolicy::ExtendedBlockKrylovIterations>(tr_rank,tol);

    rsi.compute(A);
    rsi_ext.compute(A);
    rbki.compute(A);
    rbki_ext.compute(A);

    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi_ext.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki_ext.singularValues()).lpNorm<2>() < tol);
}

TEST(rand_svd_test, rect_test){
    DMatrix<double> A = DMatrix<double>::Random(10,20);
    int tr_rank = 3; double tol = 1e-3;

    RandomizedSVD<decltype(A),IterationPolicy::SubspaceIterations> rsi(A,tr_rank);
    RandomizedSVD<decltype(A),IterationPolicy::ExtendedSubspaceIterations> rsi_ext(A,tr_rank);
    RandomizedSVD<decltype(A),IterationPolicy::BlockKrylovIterations> rbki(A,tr_rank);
    RandomizedSVD<decltype(A),IterationPolicy::ExtendedBlockKrylovIterations> rbki_ext(A,tr_rank);
    Eigen::JacobiSVD<DMatrix<double>> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi_ext.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki_ext.singularValues()).lpNorm<2>() < tol);

    rsi.compute(A.transpose());
    rsi_ext.compute(A.transpose());
    rbki.compute(A.transpose());
    rbki_ext.compute(A.transpose());
    svd.compute(A);

    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rsi_ext.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki.singularValues()).lpNorm<2>() < tol);
    EXPECT_TRUE((svd.singularValues().head(tr_rank)-rbki_ext.singularValues()).lpNorm<2>() < tol);
}

TEST(randomized_range_finders, reconstruction_accuracy){
    DMatrix<double> A = DMatrix<double>::Random(40,20);
    A = A*A.transpose();
    double tol = 1e-3;

    RandomizedRangeFinder<decltype(A),IterationPolicy::SubspaceIterations,StoppingPolicy::ReconstructionAccuracy>
            rsi_rf(A,20);
    RandomizedRangeFinder<decltype(A),IterationPolicy::ExtendedSubspaceIterations,StoppingPolicy::ReconstructionAccuracy>
            rsi_ext_rf(A,20);
    RandomizedRangeFinder<decltype(A),IterationPolicy::BlockKrylovIterations,StoppingPolicy::ReconstructionAccuracy>
            rbki_rf(A,1);
    RandomizedRangeFinder<decltype(A),IterationPolicy::ExtendedBlockKrylovIterations,StoppingPolicy::ReconstructionAccuracy>
            rbki_ext_rf(A,1);

    EXPECT_TRUE(std::pow((A-rsi_rf.rangeMatrix()*rsi_rf.residualMatrix()*rsi_rf.corangeMatrix().transpose()).norm(),2) < std::pow(tol*A.norm(),2));
    EXPECT_TRUE(std::pow((A-rsi_ext_rf.rangeMatrix()*rsi_ext_rf.residualMatrix()*rsi_ext_rf.corangeMatrix().transpose()).norm(),2) < std::pow(tol*A.norm(),2));
    EXPECT_TRUE(std::pow((A-rbki_rf.rangeMatrix()*rbki_rf.residualMatrix()*rbki_rf.corangeMatrix().transpose()).norm(),2) < std::pow(tol*A.norm(),2));
    EXPECT_TRUE(std::pow((A-rbki_ext_rf.rangeMatrix()*rbki_ext_rf.residualMatrix()*rbki_ext_rf.corangeMatrix().transpose()).norm(),2) < std::pow(tol*A.norm(),2));
}

TEST(nystrom_randomized_svd, reconstruction_accuracy){
    DMatrix<double> A = DMatrix<double>::Random(20,10);
    A = A*A.transpose();
    double tol = 1e-4;

    RandomizedEVD<decltype(A),IterationPolicy::BlockKrylovIterations> nys_rbki(A,tol);
    RandomizedEVD<decltype(A),IterationPolicy::RandomlyPivotedCholesky> rp_chol(A,tol);

    DMatrix<double> rbki_reconstruction = nys_rbki.matrixU()*nys_rbki.eigenValues().asDiagonal()*nys_rbki.matrixU().transpose();
    DMatrix<double> rpchol_reconstruction = rp_chol.matrixU()*rp_chol.eigenValues().asDiagonal()*rp_chol.matrixU().transpose();

    EXPECT_TRUE(std::pow((A-rbki_reconstruction).norm(),2) < std::pow(tol*A.norm(),2));
    EXPECT_TRUE(std::pow((A-rpchol_reconstruction).norm(),2) < std::pow(tol*A.norm(),2));

    //other constructor
    nys_rbki = RandomizedEVD<decltype(A),IterationPolicy::BlockKrylovIterations>(tol);
    rp_chol = RandomizedEVD<decltype(A),IterationPolicy::RandomlyPivotedCholesky>(tol);

    nys_rbki.compute(A);
    rp_chol.compute(A);

    rbki_reconstruction = nys_rbki.matrixU()*nys_rbki.eigenValues().asDiagonal()*nys_rbki.matrixU().transpose();
    rpchol_reconstruction = rp_chol.matrixU()*rp_chol.eigenValues().asDiagonal()*rp_chol.matrixU().transpose();

    EXPECT_TRUE(std::pow((A-rbki_reconstruction).norm(),2) < std::pow(tol*A.norm(),2));
    EXPECT_TRUE(std::pow((A-rpchol_reconstruction).norm(),2) < std::pow(tol*A.norm(),2));
}
