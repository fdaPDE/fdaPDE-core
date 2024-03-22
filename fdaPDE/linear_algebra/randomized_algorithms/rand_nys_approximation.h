//
// Created by Marco Galliani on 02/10/24.
//

#ifndef NYSTROM_APPROXIMATION_H
#define NYSTROM_APPROXIMATION_H

#include <algorithm>
#include <random>
#include <unordered_set>
#include <Eigen/Cholesky>
#include <Eigen/QR>

#include "utils/random_utils.h"
#include "fdaPDE/utils/symbols.h"

namespace fdapde{
namespace core{

template<typename MatrixType, IterationPolicy ItPolicy> class NysIterations;

//Nystrom Low-rank Approximator
template<typename MatrixType,IterationPolicy ItPolicy>
class NystromApproximator{
private:
    //method parameters
    int block_sz_;
    double shift_;
    double tolerance_=1e-3;
    int seed_=fdapde::random_seed;
    //Storage of the decomposition
    DMatrix<double> Z_; //factor matrix: A + shift*I = Z*Z^T
    //Iterator
    friend NysIterations<MatrixType,ItPolicy>;
    NysIterations<MatrixType,ItPolicy> InitNysIterations(const MatrixType &A){
        return NysIterations<MatrixType,ItPolicy>(A, this);
    };
public:
    //constructors
    NystromApproximator() = default;
    NystromApproximator(int block_sz, int seed=fdapde::random_seed) : block_sz_(block_sz), seed_(seed){}
    //computation
    NystromApproximator compute(const MatrixType &A){
        for(auto rf_it=InitNysIterations(A); !rf_it.stop(); ++rf_it){}
        return *this;
    }
    //setters
    inline void setBlockSize(int block_sz){ block_sz_ = block_sz;}
    inline void setSeed(int seed){ seed_ = seed;}
    inline void setTol(double tol){ tolerance_ = tol;}

    //getters
    inline DMatrix<double>& matrixF(){ return Z_;}
    inline double shift(){ return shift_;}
    //freeing the space occupied by the storage of the decompositions
    void flush(){
        Z_.resize(0,0);
    }
};

//NysBKI: Nystrom Block Krylov Iterations
//assumption -> A is sdp
template<typename MatrixType>
class NysIterations<MatrixType,IterationPolicy::BlockKrylovIterations>{
private:
    int index_,maxIter_;
    DMatrix<double> A_; //matrix to be decomposed
    double norm_A_;
    //Storage of the decomposition
    DMatrix<double> X_; //range and corange
    DMatrix<double> T_; //core matrix
    //Method parameters
    size_t block_sz_;
    double shift_;
    //Pointer to the calling Range Finder
    NystromApproximator<MatrixType,IterationPolicy::BlockKrylovIterations> *nys_approx_;
public:
    NysIterations(const MatrixType &A, NystromApproximator<MatrixType,IterationPolicy::BlockKrylovIterations>* nys_approx) :
    A_(A),nys_approx_(nys_approx){
        index_=0;
        block_sz_ = nys_approx_->block_sz_;
        //init matrices dimensions
        maxIter_ = std::ceil((double)A.rows()/(double)block_sz_); //A.rows() is equal to A.cols()
        auto maxMatDim = maxIter_*block_sz_; //A.rows() is equal to A.cols()
        //The Krylov subspace is used to approximate the range of A, it is stored in Q_
        X_.resize(A_.rows(), maxMatDim);
        //Residual matrix
        T_.resize(A.rows(), maxMatDim);
        //Init T_ and shift_ parameter
        T_.leftCols(block_sz_) = GaussianMatrix(A.rows(), block_sz_, nys_approx_->seed_);
        shift_ = DOUBLE_TOLERANCE*A_.trace();
        //stopping criterion
        norm_A_ = A_.norm();
    }
    bool stop(){
        if(index_==0) return false;
        double squared_reconstr_err = std::pow(norm_A_,2)-std::pow((X_.leftCols(index_*block_sz_)*T_.leftCols(index_*block_sz_).transpose()).norm(),2);
        return squared_reconstr_err < std::pow(norm_A_*nys_approx_->tolerance_,2) || index_>=maxIter_;
    }
    NysIterations& operator++(){
        Eigen::HouseholderQR<DMatrix<double>> qr;
        X_.middleCols(index_*block_sz_,block_sz_) = T_.middleCols(std::max(index_-1,0)*block_sz_,block_sz_);
        //Block Gram-Schmidt
        X_.middleCols(index_*block_sz_,block_sz_) =
                (DMatrix<double>::Identity(A_.rows(),A_.rows()) - X_.leftCols(index_*block_sz_) * X_.leftCols(index_*block_sz_).transpose()) * X_.middleCols(index_*block_sz_,block_sz_);
        X_.middleCols(index_*block_sz_,block_sz_) =
                (DMatrix<double>::Identity(A_.rows(),A_.rows()) - X_.leftCols(index_*block_sz_) * X_.leftCols(index_*block_sz_).transpose()) * X_.middleCols(index_*block_sz_,block_sz_);
        qr.compute(X_.middleCols(index_*block_sz_, block_sz_));
        X_.middleCols(index_*block_sz_, block_sz_) = qr.householderQ() * DMatrix<double>::Identity(A_.rows(),block_sz_);
        //Updating the residual matrix
        T_.middleCols(index_*block_sz_,block_sz_) = (A_+shift_*DMatrix<double>::Identity(A_.rows(),A_.cols()))*X_.middleCols(index_*block_sz_, block_sz_);

        index_++;
        return *this;
    }
    ~NysIterations(){
        Eigen::LLT<DMatrix<double>> chol = Eigen::LLT<DMatrix<double>>(X_.leftCols(index_*block_sz_).transpose()*T_.leftCols(index_*block_sz_));
        nys_approx_->Z_ = chol.matrixU().solve<Eigen::OnTheRight>(T_.leftCols(index_*block_sz_));
        nys_approx_->shift_ = shift_; //passing the shift parameter (computed on A)
    }
};

//Randomly pivoted Cholesky
template<typename MatrixType>
class NysIterations<MatrixType,IterationPolicy::RandomlyPivotedCholesky>{
private:
    int index_,col_idx_,maxIter_;
    DMatrix<double> A_; //matrix to be decomposed
    double norm_A_;
    std::vector<int> ind_colsA_;
    //Storage of the decomposition
    DMatrix<double> Z_;
    DVector<double> d_; //sampling distribution
    //Method parameters
    size_t block_sz_;
    double shift_;
    //Random engine
    std::mt19937 rand_gen_;
    //Pointer to the calling Range Finder
    NystromApproximator<MatrixType,IterationPolicy::RandomlyPivotedCholesky> *nys_approx_;
public:
    NysIterations(const MatrixType &A, NystromApproximator<MatrixType,IterationPolicy::RandomlyPivotedCholesky>* nys_approx) :
    A_(A),rand_gen_(nys_approx->seed_),nys_approx_(nys_approx){
        index_=0; col_idx_=0;
        block_sz_ = nys_approx_->block_sz_;
        //init matrices dimensions
        A_ = A; norm_A_=A.norm();
        ind_colsA_ = std::vector<int>(A.cols());
        std::iota(ind_colsA_.begin(), ind_colsA_.end(), 0);
        Z_ = DMatrix<double>::Zero(A.rows(),A.cols());
        d_ = A.diagonal();
        maxIter_ = std::floor((double)A.cols()/(double)block_sz_);
    }
    bool stop(){
        return index_ >= maxIter_ || (A_-Z_.leftCols(col_idx_)*Z_.leftCols(col_idx_).transpose()).norm() < nys_approx_->tolerance_*norm_A_;
    }
    NysIterations& operator++(){
        //sampling the columns
        std::discrete_distribution<int> sampling_distr(d_.begin(),d_.end());
        std::unordered_set<int> sampled_pivots(block_sz_);
        while(sampled_pivots.size()<block_sz_) sampled_pivots.insert(sampling_distr(rand_gen_));
        std::vector<int> pivot_set(sampled_pivots.begin(),sampled_pivots.end()); //converting for Eigen slicing
        //build the F factor
        DMatrix<double> G = A_(Eigen::all,pivot_set);
        G = G - Z_.leftCols(index_*block_sz_) * Z_(pivot_set,Eigen::all).leftCols(index_*block_sz_).transpose();
        shift_ = DOUBLE_TOLERANCE*G(pivot_set,Eigen::all).trace();
        Eigen::LLT<DMatrix<double>> chol(G(pivot_set,Eigen::all) + shift_*DMatrix<double>::Identity(block_sz_,block_sz_));
        DMatrix<double> T = chol.matrixU().solve<Eigen::OnTheRight>(G);
        Z_.middleCols(index_*block_sz_,block_sz_) = T;

        //update the sampling distribution
        d_ = (d_ - T.rowwise().squaredNorm()).array().max(0);
        index_++; col_idx_+=block_sz_;
        return *this;
    }
    ~NysIterations(){
        nys_approx_->Z_ = Z_.leftCols(col_idx_);
        nys_approx_->shift_ = shift_; //passing the shift parameter (computed on A)
    }
};

/*
//NysSI: Nystrom Subspace Iterations
template<typename MatrixType, StoppingPolicy StopPolicy>
class NysIterations<MatrixType,IterationPolicy::SubspaceIterations,StopPolicy>{
private:
    int index_,maxIter_;
    DMatrix<double> A_; //matrix to be decomposed
    //Storage of the decomposition
    DMatrix<double> X_; //range
    DMatrix<double> T_; //residual matrix
    //Structured storage
    DMatrix<double> Z_; //factor: A + shift*I = Z*Z^T, alternative storage
    //Method parameters
    size_t block_sz_;
    double shift_;
    //Pointer to the calling Range Finder
    NystromApproximator<MatrixType,IterationPolicy::SubspaceIterations,StopPolicy> *nys_approx_;
public:
    NysIterations(const MatrixType &A, NystromApproximator<MatrixType,IterationPolicy::SubspaceIterations,StopPolicy>* nys_approx) :
    A_(A),nys_approx_(nys_approx){
        index_=0;
        block_sz_ = nys_approx_->block_sz_;
        //Init X_ and shift_ parameter
        T_ = GaussianMatrix(A.cols(), block_sz_, nys_approx_->seed_);
        shift_ = std::numeric_limits<float>::denorm_min()*A_.trace();
    }
    bool stop(){
        if(index_==0) return false;
        if constexpr(StopPolicy==StoppingPolicy::ReconstructionAccuracy){
            return (A_-X_*T_.transpose()).norm() < nys_approx_->tolerance_;
        }
    }
    NysIterations& operator++(){
        this->index_++;
        Eigen::HouseholderQR<decltype(T_)> qr(T_);
        X_ = qr.householderQ()*DMatrix<double>::Identity(A_.cols(), block_sz_);
        T_ = A_ * X_;
        return *this;
    }
    ~NysIterations(){
        T_ = T_ + shift_*X_;
        auto C = Eigen::LLT<DMatrix<double>>(X_.transpose()*T_).matrixL();
        nys_approx_->Z_ = C.solve<Eigen::OnTheRight>(T_);
        nys_approx_->shift_ = shift_; //passing the shift parameter (computed on A)
    }
};
*/

}//core
}//fdapde

#endif //NYSTROM_APPROXIMATION_H
