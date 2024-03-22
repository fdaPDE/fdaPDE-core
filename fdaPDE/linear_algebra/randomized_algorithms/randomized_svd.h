//
// Created by Marco Galliani on 19/07/24.
//
#ifndef RSVD_RANDSVD_H
#define RSVD_RANDSVD_H


#include "fdaPDE/utils/symbols.h"
#include "rand_range_finder.h"
#include "rand_nys_approximation.h"
#include <chrono>

namespace fdapde{
namespace core{

template<typename MatrixType, IterationPolicy ItPolicy=ExtendedSubspaceIterations>
class RandomizedSVD{
private:
    RandomizedRangeFinder<MatrixType,ItPolicy,StoppingPolicy::SingularValuesAccuracy> rf_;
    int tr_rank_;
    DMatrix<double> U_, V_;
    DVector<double> Sigma_;
public:
    RandomizedSVD(int tr_rank, double tol=1e-3, int seed=fdapde::random_seed) : tr_rank_(tr_rank){
        rf_.setSeed(seed);
        rf_.setNSingVecs(tr_rank);
        rf_.setTol(tol);
    }
    RandomizedSVD(const MatrixType &A, int tr_rank, int block_sz, double tol=1e-3, int seed=fdapde::random_seed) : tr_rank_(tr_rank){
        rf_.setSeed(seed);
        rf_.setNSingVecs(tr_rank);
        rf_.setTol(tol);
        compute(A,block_sz);
    }
    RandomizedSVD(const MatrixType &A, int tr_rank) : tr_rank_(tr_rank){
        rf_.setNSingVecs(tr_rank);
        rf_.setTol(1e-3);
        rf_.setSeed(fdapde::random_seed);
        compute(A);
    }
    RandomizedSVD& compute(const MatrixType &A){
        //setting the block size to a default value depending on the algorithm
        int max_rank = std::min(A.rows(),A.cols()), block_sz;
        if constexpr (ItPolicy==IterationPolicy::SubspaceIterations || ItPolicy==IterationPolicy::ExtendedSubspaceIterations){
            block_sz = std::min(2*tr_rank_,max_rank);
        }else{
            if(max_rank < 100){
                block_sz = 1;
            }else{
                block_sz = 10;
            }
        }
        return compute(A,block_sz);
    }
    RandomizedSVD& compute(const MatrixType &A, int block_sz){
        rf_.setBlockSize(block_sz);
        //computation of the range
        rf_.compute(A);
        //construction of the svd
        Eigen::JacobiSVD<DMatrix<double>> svd = rf_.residualMatrixSVD();;
        int est_rank = svd.singularValues().size();
        U_ = rf_.rangeMatrix()*svd.matrixU().leftCols(std::min(tr_rank_,est_rank));
        V_ = rf_.corangeMatrix()*svd.matrixV().leftCols(std::min(tr_rank_,est_rank));
        Sigma_ = svd.singularValues().head(std::min(tr_rank_,est_rank));
        if(est_rank < tr_rank_) tr_rank_=est_rank;
        //freeing the space occupied by the low-rank decomposition (the decomposition is now stored as an SVD)
        rf_.flush();
        return *this;
    }
    inline int rank() const{ return tr_rank_;}
    inline const DMatrix<double> &matrixU() const{ return U_;}
    inline const DMatrix<double> &matrixV() const{ return V_;}
    inline const DVector<double> &singularValues() const{ return Sigma_; }
};

template<typename MatrixType, IterationPolicy ItPolicy>
class RandomizedEVD{
private:
    NystromApproximator<MatrixType,ItPolicy> nys_approx_;
    DMatrix<double> U_;
    DVector<double> Lambda_;
    int rank_;
public:
    RandomizedEVD(double tol=1e-3, int seed=fdapde::random_seed){
        nys_approx_.setTol(tol);
        nys_approx_.setSeed(seed);
    }
    RandomizedEVD(const MatrixType &A, double tol=1e-3, int seed=fdapde::random_seed){
        nys_approx_.setTol(tol);
        nys_approx_.setSeed(seed);
        compute(A);
    }
    RandomizedEVD(const MatrixType &A, int block_size, double tol = 1e-3, int seed=fdapde::random_seed) : nys_approx_(block_size,seed){
        nys_approx_.setTol(tol);
        nys_approx_.setSeed(seed);
        compute(A,block_size);
    }
    RandomizedEVD& compute(const MatrixType &A){
        int block_sz;
        if(A.cols() < 100){
            block_sz = 1;
        }
        else{
            block_sz = 10;
        }
        return compute(A,block_sz);
    }
    RandomizedEVD& compute(const MatrixType &A, int block_sz){
        //Computing a Nystrom approximation
        nys_approx_.setBlockSize(block_sz);
        nys_approx_.compute(A);
        //Construct the SVD
        Eigen::JacobiSVD<DMatrix<double>> svd(nys_approx_.matrixF(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        U_ = svd.matrixU();
        Lambda_ = (svd.singularValues().array().pow(2)-nys_approx_.shift()*Eigen::ArrayXd::Ones(svd.singularValues().size())).max(0);
        rank_ = Lambda_.size();
        //freeing the space occupied by the low-rank decomposition (the decomposition is now stored as an Eigen Decomposition)
        nys_approx_.flush();
        return *this;
    }
    inline DMatrix<double> &matrixU(){ return U_;}
    inline DVector<double> &eigenValues(){ return Lambda_; }
    inline int rank() const{ return rank_; }
};

}//namespace core
}//namespace fdapde

#endif //RSVD_RANDSVD_H
