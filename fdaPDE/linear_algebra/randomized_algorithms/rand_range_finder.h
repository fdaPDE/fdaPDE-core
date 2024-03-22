//
// Created by Marco Galliani on 19/07/24.
//

#ifndef RAND_RANGE_FINDER_H
#define RAND_RANGE_FINDER_H

#include <type_traits>
#include <limits>

#include "utils/random_utils.h"

namespace fdapde{
namespace core{

//Randomized Range Finder
template<typename MatrixType, IterationPolicy ItPolicy, StoppingPolicy StopPolicy> class RFIterations;

template<typename MatrixType, IterationPolicy ItPolicy, StoppingPolicy StopPolicy>
class RandomizedRangeFinder{
private:
    //method parameters
    int block_sz_;
    int seed_=fdapde::random_seed;
    double tolerance_=1e-3;
    int n_sing_vecs_=1;
    int maxIter_=1e4;

    //Storage of the decomposition
    DMatrix<double> X_, Y_; //range and corange
    DMatrix<double> T_; //core matrix
    Eigen::JacobiSVD<DMatrix<double>> svd_; //svd device used to decompose T_

    //Iterations
    friend RFIterations<MatrixType,ItPolicy,StopPolicy>;
    RFIterations<MatrixType,ItPolicy,StopPolicy> InitRFIterations(const MatrixType &A){
        return RFIterations<MatrixType,ItPolicy,StopPolicy>(A,this);
    };
public:
    //constructors
    RandomizedRangeFinder() = default;
    RandomizedRangeFinder(int block_sz, int seed=fdapde::random_seed) : block_sz_(block_sz),seed_(seed){}
    RandomizedRangeFinder(const MatrixType &A, int block_sz, int seed=fdapde::random_seed) : block_sz_(block_sz),seed_(seed){
        compute(A);
    }
    //computation
    RandomizedRangeFinder& compute(const MatrixType &A){
        for(auto rf_it=this->InitRFIterations(A); !rf_it.stop(); ++rf_it){}
        return *this;
    }
    //setters
    inline void setBlockSize(int block_sz){ block_sz_=block_sz;}
    inline void setSeed(int seed){ seed_=seed;}
    inline void setTol(double tol){ tolerance_ = tol;}
    inline void setMaxIt(int maxIt){ maxIter_ = maxIt;}
    inline void setNSingVecs(int n_sing_vecs){ n_sing_vecs_ = n_sing_vecs;}

    //getters
    inline const DMatrix<double> &residualMatrix() const{ return T_;}
    inline const DMatrix<double> &rangeMatrix() const{ return X_;}
    inline const DMatrix<double> &corangeMatrix() const{ return Y_;}
    Eigen::JacobiSVD<DMatrix<double>> &residualMatrixSVD(){
        if(svd_.cols()==-1){
            svd_.compute(T_,Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        return svd_;
    }
    //freeing the space occupied by the storage of the decompositions
    void flush(){
        X_.resize(0,0); Y_.resize(0,0); T_.resize(0,0);
        svd_ = Eigen::JacobiSVD<DMatrix<double>>();
    }
};

//SubspaceIterations
template<typename MatrixType, StoppingPolicy StopPolicy>
class RFIterations<MatrixType,IterationPolicy::SubspaceIterations,StopPolicy>{
private:
    MatrixType A_; //matrix to be decomposed
    double norm_A_;
    //Storage of the decomposition
    DMatrix<double> X_, Y_; //range and corange
    DMatrix<double> T_; //core matrix
    //Method parameters
    int block_sz_;
    //Iterations
    int index_;
    //Pointer to the calling Range Finder
    RandomizedRangeFinder<MatrixType,IterationPolicy::SubspaceIterations,StopPolicy> *rf_;
public:
    RFIterations(const MatrixType &A, RandomizedRangeFinder<MatrixType,IterationPolicy::SubspaceIterations,StopPolicy>* rf) : A_(A), rf_(rf){
        //init the params
        index_=0;
        block_sz_ = rf_->block_sz_;
        norm_A_=A_.norm();
        //Init X_
        Eigen::HouseholderQR<DMatrix<double>> qr(A*GaussianMatrix(A.cols(), block_sz_, rf_->seed_));
        X_ = qr.householderQ()*DMatrix<double>::Identity(A_.rows(),block_sz_);
        T_ = A.transpose()*X_;
    }
    bool stop(){
        if(index_==0) return false;
        if constexpr (StopPolicy==StoppingPolicy::ReconstructionAccuracy){
            auto squared_reconstr_err = std::pow(norm_A_,2) - std::pow((X_*T_.transpose()).norm(),2);
            return (squared_reconstr_err < std::pow(rf_->tolerance_*norm_A_,2) || index_>=rf_->maxIter_);
        }else if constexpr (StopPolicy==StoppingPolicy::SingularValuesAccuracy){
            rf_->svd_.compute(T_.transpose(),Eigen::ComputeThinU | Eigen::ComputeThinV);
            auto E = A_*rf_->svd_.matrixV().leftCols(rf_->n_sing_vecs_)-X_*rf_->svd_.matrixU().leftCols(rf_->n_sing_vecs_)*rf_->svd_.singularValues().head(rf_->n_sing_vecs_).asDiagonal();
            return E.colwise().template lpNorm<2>().maxCoeff() < rf_->tolerance_ || index_>=rf_->maxIter_;
        }
    };
    RFIterations& operator++(){
        Eigen::HouseholderQR<DMatrix<double>> qr(T_);
        X_ = qr.householderQ()*DMatrix<double>::Identity(A_.cols(), block_sz_);
        T_ = A_ * X_;
        qr.compute(T_);
        X_ = qr.householderQ()*DMatrix<double>::Identity(A_.rows(), block_sz_);
        T_ = A_.transpose()*X_;
        this->index_++;
        return *this;
    }
    ~RFIterations(){
        rf_->X_ = X_; rf_->T_ = T_.transpose();
        //dummy init: corange not approximated by this method
        rf_->Y_ = DMatrix<double>::Identity(A_.cols(),A_.cols());
    }
};

//ExtendedSubspaceIterations
template<typename MatrixType, StoppingPolicy StopPolicy>
class RFIterations<MatrixType,IterationPolicy::ExtendedSubspaceIterations,StopPolicy>{
private:
    MatrixType A_; //matrix to be decomposed
    double norm_A_;
    //Storage of the decomposition
    DMatrix<double> X_, Y_; //range and corange
    DMatrix<double> T_; //core matrix
    //Method parameters
    int block_sz_;
    //Iterations
    int index_;
    //Pointer to the calling Range Finder
    RandomizedRangeFinder<MatrixType,IterationPolicy::ExtendedSubspaceIterations,StopPolicy> *rf_;
public:
    RFIterations(const MatrixType &A, RandomizedRangeFinder<MatrixType,IterationPolicy::ExtendedSubspaceIterations,StopPolicy>* rf) : A_(A), rf_(rf){
        //init the params
        index_=0;
        block_sz_ = rf_->block_sz_;
        norm_A_=A.norm();
        //Init X_
        Eigen::HouseholderQR<DMatrix<double>> qr(A*GaussianMatrix(A.cols(), block_sz_, rf_->seed_));
        X_ = qr.householderQ()*DMatrix<double>::Identity(A_.rows(),block_sz_);
    }
    bool stop(){
        if(index_==0) return false;
        if constexpr (StopPolicy==StoppingPolicy::ReconstructionAccuracy){
            double squared_reconstr_err = std::pow(norm_A_,2) - std::pow((X_*T_*Y_.transpose()).norm(),2);
            return (squared_reconstr_err < std::pow(rf_->tolerance_*norm_A_,2) || index_>=rf_->maxIter_);
        }else if constexpr (StopPolicy==StoppingPolicy::SingularValuesAccuracy){
            rf_->svd_.compute(T_,Eigen::ComputeThinU | Eigen::ComputeThinV);
            DMatrix<double> E;
            if((this->index_-1)%2==0){
                E = A_*Y_*rf_->svd_.matrixV().leftCols(rf_->n_sing_vecs_) - X_*rf_->svd_.matrixU().leftCols(rf_->n_sing_vecs_)*rf_->svd_.singularValues().head(rf_->n_sing_vecs_).asDiagonal();
            }else{
                E = A_.transpose()*X_*rf_->svd_.matrixU().leftCols(rf_->n_sing_vecs_) - Y_*rf_->svd_.matrixV().leftCols(rf_->n_sing_vecs_)*rf_->svd_.singularValues().head(rf_->n_sing_vecs_).asDiagonal();
            }
            return E.colwise().template lpNorm<2>().maxCoeff() < rf_->tolerance_ || index_>rf_->maxIter_;
        }
    };
    RFIterations& operator++(){
        if(index_%2 == 0){
            Y_= A_.transpose() * X_;
            Eigen::HouseholderQR<DMatrix<double>> qr(Y_);
            Y_ = qr.householderQ()*DMatrix<double>::Identity(Y_.rows(),block_sz_);
            T_ = qr.matrixQR().triangularView<Eigen::Upper>();
            DMatrix<double> tmp = T_.topRows(block_sz_).transpose();
            T_ = tmp;
        }else{
            X_ = A_ *Y_;
            Eigen::HouseholderQR<DMatrix<double>> qr(X_);
            X_ = qr.householderQ()*DMatrix<double>::Identity(X_.rows(),block_sz_);
            T_ = qr.matrixQR().triangularView<Eigen::Upper>();
            DMatrix<double> tmp = T_.topRows(block_sz_);
            T_ = tmp;
        }
        this->index_++;
        return *this;
    }
    ~RFIterations(){
        rf_->X_ = X_; rf_->T_ = T_; rf_->Y_ = Y_;
    }
};

//BlockKrylovIterations
template<typename MatrixType,StoppingPolicy StopPolicy>
class RFIterations<MatrixType,IterationPolicy::BlockKrylovIterations,StopPolicy>{
private:
    MatrixType A_; //matrix to be decomposed
    double norm_A_;
    //Storage of the decomposition
    DMatrix<double> X_, Y_; //range and corange
    DMatrix<double> T_; //core matrix
    //Method parameters
    int block_sz_;
    //Iterations
    int index_;
    //Pointer to the calling Range Finder
    RandomizedRangeFinder<MatrixType,IterationPolicy::BlockKrylovIterations,StopPolicy> *rf_;
public:
    RFIterations(const MatrixType &A, RandomizedRangeFinder<MatrixType,IterationPolicy::BlockKrylovIterations,StopPolicy>* rf) : A_(A), rf_(rf){
        index_=0;
        block_sz_ = rf_->block_sz_;
        norm_A_ = A_.norm();
        int maxMatDim = std::ceil((double)std::min(A_.rows(), A_.cols())/(double)block_sz_)*block_sz_;
        rf_->maxIter_ = std::min(rf_->maxIter_, static_cast<int>(maxMatDim/block_sz_));

        Eigen::HouseholderQR<DMatrix<double>> qr;
        //The Krylov subspace is used to approximate the range of A, it is stored in Q_
        X_.resize(A_.rows(), maxMatDim);
        X_.leftCols(block_sz_) = A_ * GaussianMatrix(A.cols(), block_sz_, rf_->seed_);
        qr.compute(X_.leftCols(block_sz_));
        X_.leftCols(block_sz_) = qr.householderQ() * DMatrix<double>::Identity(A_.rows(), block_sz_);
        //matrix T^T = A^T*X: later on will be used to compute A_hat = X*T=X*X^T*A (projection of A over the subspace)
        T_.resize(A.cols(), maxMatDim);
        T_.leftCols(block_sz_) = A.transpose() * X_.leftCols(block_sz_);
    }
    bool stop(){
        if(index_==0) return false;
        if constexpr (StopPolicy==StoppingPolicy::ReconstructionAccuracy){
            double squared_reconstr_err = std::pow(norm_A_,2) -std::pow((X_.leftCols(index_*block_sz_) * T_.leftCols(index_*block_sz_).transpose()).norm(),2);
            return squared_reconstr_err < std::pow(rf_->tolerance_*norm_A_,2) || index_>=rf_->maxIter_;
        }else if constexpr (StopPolicy==StoppingPolicy::SingularValuesAccuracy){
            int i = index_, N_sing_vecs = rf_->n_sing_vecs_;
            rf_->svd_.compute(T_.leftCols((i+1)*block_sz_).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            //residuals
            auto E = A_ * rf_->svd_.matrixV().leftCols(std::min(N_sing_vecs,(i+1)*block_sz_)) - X_.leftCols((i+1)*block_sz_) * rf_->svd_.matrixU().leftCols(std::min(N_sing_vecs,(i+1)*block_sz_))*rf_->svd_.singularValues().head(std::min(N_sing_vecs,(i+1)*block_sz_)).asDiagonal();
            return E.colwise().template lpNorm<2>().maxCoeff() < rf_->tolerance_ || index_>=rf_->maxIter_;
        }
    };
    RFIterations& operator++(){
        int i = index_;
        Eigen::HouseholderQR<DMatrix<double>> qr;
        X_.middleCols((i+1)*block_sz_,block_sz_) = A_ * T_.middleCols(i*block_sz_, block_sz_);

        //Implementation of BCGS+ composed with Householder
        //orthogonalization of Krylov subspace w.r.t. past iterates (SKELETON)
        X_.middleCols((i+1)*block_sz_,block_sz_) =
            (DMatrix<double>::Identity(A_.rows(),A_.rows()) - X_.leftCols((i+1)*block_sz_) * X_.leftCols((i+1)*block_sz_).transpose()) * X_.middleCols((i+1)*block_sz_,block_sz_);
        //orthogonalization of the block (MUSCLE)
        qr.compute(X_.middleCols((i+1)*block_sz_, block_sz_));
        X_.middleCols((i+1)*block_sz_, block_sz_) = qr.householderQ() * DMatrix<double>::Identity(A_.rows(),block_sz_);
        //orthogonalization of Krylov subspace w.r.t. past iterates (SKELETON)
        X_.middleCols((i+1)*block_sz_,block_sz_) =
            (DMatrix<double>::Identity(A_.rows(),A_.rows()) - X_.leftCols((i+1)*block_sz_) * X_.leftCols((i+1)*block_sz_).transpose()) * X_.middleCols((i+1)*block_sz_,block_sz_);
        //orthogonalization of the block (MUSCLE)
        qr.compute(X_.middleCols((i+1)*block_sz_, block_sz_));
        X_.middleCols((i+1)*block_sz_, block_sz_) = qr.householderQ() * DMatrix<double>::Identity(A_.rows(),block_sz_);

        //projection of A over the new block
        T_.middleCols((i+1)*block_sz_,block_sz_) = A_.transpose() * X_.middleCols((i+1)*block_sz_,block_sz_);
        index_++;
        return *this;
    }
    ~RFIterations(){
        rf_->X_ = X_.leftCols((index_+1)*block_sz_);
        rf_->T_ = T_.leftCols((index_+1)*block_sz_).transpose();
        rf_->Y_ = DMatrix<double>::Identity(A_.cols(),A_.cols());
    }
};

//ExtendedBlockKrylovIterations
template<typename MatrixType, StoppingPolicy StopPolicy>
class RFIterations<MatrixType,IterationPolicy::ExtendedBlockKrylovIterations,StopPolicy>{
private:
    DMatrix<double> A_; //matrix to be decomposed
    double norm_A_;
    //Storage of the decomposition
    DMatrix<double> X_, Y_; //range and corange
    int sizeX_,sizeY_;
    DMatrix<double> T_; //core matrix
    //Support for the computations
    DMatrix<double> R_, blockR_, colR_; //matrices used to iteratively update R (meaning T)
    DMatrix<double> S_, blockS_, colS_; //matrices used to iteratively update S (meaning T)
    //For the computation of the error on the singular values
    DMatrix<double> Z_, W_; //storing forward unorthogonalized range and corange
    //Method parameters
    int block_sz_;
    //Pointer to the calling Range Finder
    RandomizedRangeFinder<MatrixType,IterationPolicy::ExtendedBlockKrylovIterations,StopPolicy>* rf_;
    int index_;
    //auxiliary index for taking into account the half-iterations
    int aux_index_;
public:
    RFIterations(const MatrixType &A, RandomizedRangeFinder<MatrixType,IterationPolicy::ExtendedBlockKrylovIterations,StopPolicy>* rf) : A_(A), rf_(rf){
        index_ = 0; aux_index_=0;
        block_sz_ = rf_->block_sz_;
        norm_A_=A.norm();
        //ASSUMPTION: qk < std::min(A.rows(), A.cols()), X.cols() = qk, Y.cols() = qk
        int maxMatDim = std::ceil((double)std::min(A_.rows(), A_.cols())/(double)block_sz_)*block_sz_;
        rf_->maxIter_ = std::min(rf_->maxIter_, static_cast<int>(maxMatDim/block_sz_));

        //Initialising matrix dimensions
        X_.resize(A_.rows(),maxMatDim); Z_.resize(A_.rows(),maxMatDim);
        Y_.resize(A_.cols(),maxMatDim); W_.resize(A_.cols(),maxMatDim);
        R_.resize(maxMatDim,maxMatDim); S_.resize(maxMatDim+block_sz_,maxMatDim);
        //Initialise matrix for the computations
        Eigen::HouseholderQR<DMatrix<double>> qr;
        X_.leftCols(block_sz_) = A * GaussianMatrix(A.cols(),block_sz_,rf_->seed_);
        qr.compute(X_.leftCols(block_sz_));
        X_.leftCols(block_sz_) = qr.householderQ() * DMatrix<double>::Identity(A.rows(),block_sz_);
        W_.leftCols(block_sz_) = A.transpose() * X_.leftCols(block_sz_);
    }
    bool stop(){
        if(index_==0) return false;
        if constexpr (StopPolicy==StoppingPolicy::ReconstructionAccuracy){
            auto A_hat = X_.leftCols(sizeX_) * T_ * Y_.leftCols(sizeY_).transpose();
            double squared_recontr_err = std::pow(norm_A_,2)-std::pow(A_hat.norm(),2);
            return squared_recontr_err < std::pow(rf_->tolerance_*norm_A_,2) || index_>rf_->maxIter_;
        }else if constexpr (StopPolicy==StoppingPolicy::SingularValuesAccuracy){
            int i = index_-1; //step back from the update in operator()++
            rf_->svd_.compute(T_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            DMatrix<double> E;
            if(i%2==0){
                E = Z_.leftCols(sizeX_)*rf_->svd_.matrixV().leftCols(std::min(rf_->n_sing_vecs_,sizeX_)) - X_.leftCols(sizeX_)*(rf_->svd_.matrixU().leftCols(std::min(rf_->n_sing_vecs_,sizeX_)))*rf_->svd_.singularValues().head(std::min(rf_->n_sing_vecs_,sizeX_)).asDiagonal();
            }else{
                E = W_.leftCols(sizeX_)*rf_->svd_.matrixU().leftCols(std::min(rf_->n_sing_vecs_,sizeY_)) - Y_.leftCols(sizeY_)*rf_->svd_.matrixV().leftCols(std::min(rf_->n_sing_vecs_,sizeY_))*rf_->svd_.singularValues().head(std::min(rf_->n_sing_vecs_,sizeY_)).asDiagonal();
            }
            auto err = E.colwise().template lpNorm<2>().maxCoeff();
            return err < rf_->tolerance_ || aux_index_>rf_->maxIter_;
        }
    };

    RFIterations& operator++(){
        if(index_%2 == 0){
            int j = aux_index_ = index_/2;
            Y_.middleCols(j*block_sz_,block_sz_) = W_.middleCols(j*block_sz_,block_sz_);
            colR_ = Y_.leftCols(j*block_sz_).transpose() * Y_.middleCols(j*block_sz_,block_sz_);
            //orthogonalization of the block matrix via Block-Gram-Schmidt (BGS)
            //MUSCLE: householder QR
            Eigen::HouseholderQR<DMatrix<double>> qr;
            //SKELETON: BCGS+
            Y_.middleCols(j*block_sz_,block_sz_) =
                (DMatrix<double>::Identity(A_.cols(),A_.cols()) - Y_.leftCols(j*block_sz_)*Y_.leftCols(j*block_sz_).transpose())*Y_.middleCols(j*block_sz_,block_sz_);
            Y_.middleCols(j*block_sz_,block_sz_) =
                (DMatrix<double>::Identity(A_.cols(),A_.cols()) - Y_.leftCols(j*block_sz_)*Y_.leftCols(j*block_sz_).transpose())*Y_.middleCols(j*block_sz_,block_sz_);
            qr.compute(Y_.middleCols(j*block_sz_,block_sz_));
            Y_.middleCols(j*block_sz_,block_sz_) = qr.householderQ() * DMatrix<double>::Identity(A_.cols(),block_sz_);
            //assembling the R matrix
            blockR_ = qr.matrixQR().triangularView<Eigen::Upper>().toDenseMatrix();
            R_.block(0,j*block_sz_,colR_.rows(),block_sz_) = colR_;
            R_.block(colR_.rows(),j*block_sz_,block_sz_,block_sz_) = blockR_.topRows(block_sz_);
            //updating Z and T
            Z_.middleCols(j*block_sz_,block_sz_) = A_ * Y_.middleCols(j*block_sz_,block_sz_);
            T_ = R_.block(0,0,(j+1)*block_sz_,(j+1)*block_sz_).triangularView<Eigen::Upper>().toDenseMatrix().transpose();
            //updating dimensions
            sizeX_ = (j+1)*block_sz_;
            sizeY_ = (j+1)*block_sz_;
        }else{
            int j = aux_index_ = (index_+1)/2;
            X_.middleCols(j*block_sz_,block_sz_) = Z_.middleCols((j-1)*block_sz_,block_sz_);
            colS_ = X_.leftCols(j*block_sz_).transpose() * X_.middleCols(j*block_sz_,block_sz_);
            //orthogonalization with Block-Gram-Schmidt (BGS)
            //MUSCLE: householder QR
            Eigen::HouseholderQR<DMatrix<double>> qr;
            //SKELETON: BCGS+
            X_.middleCols(j*block_sz_,block_sz_) =
                    (DMatrix<double>::Identity(A_.rows(),A_.rows()) - X_.leftCols(j*block_sz_)*X_.leftCols(j*block_sz_).transpose())*X_.middleCols(j*block_sz_,block_sz_);
            X_.middleCols(j*block_sz_,block_sz_) =
                    (DMatrix<double>::Identity(A_.rows(),A_.rows()) - X_.leftCols(j*block_sz_)*X_.leftCols(j*block_sz_).transpose())*X_.middleCols(j*block_sz_,block_sz_);
            qr.compute(X_.middleCols(j*block_sz_,block_sz_));
            X_.middleCols(j*block_sz_,block_sz_) = qr.householderQ() * DMatrix<double>::Identity(A_.rows(),block_sz_);
            //assembling the S matrix
            blockS_ = qr.matrixQR().triangularView<Eigen::Upper>().toDenseMatrix();
            S_.block(0,(j-1)*block_sz_,colS_.rows(),block_sz_) = colS_;
            S_.block(colS_.rows(), (j-1)*block_sz_,block_sz_,block_sz_) = blockS_.topRows(block_sz_);
            //updating W matrix and T
            W_.middleCols(j*block_sz_,block_sz_) = A_.transpose() * X_.middleCols(j*block_sz_,block_sz_);
            T_.resize((j+1)*block_sz_,j*block_sz_);
            T_ << S_.block(0,0,block_sz_,j*block_sz_), S_.block(block_sz_,0,j*block_sz_, j*block_sz_).triangularView<Eigen::Upper>().toDenseMatrix();
            //updating dimensions
            sizeX_ = (j+1)*block_sz_;
            sizeY_ = j*block_sz_;
        };
        index_++;
        return *this;
    }
    ~RFIterations(){
        rf_->X_ = X_.leftCols(sizeX_);
        rf_->T_ = T_;
        rf_->Y_ = Y_.leftCols(sizeY_);
    }
};
}//namespace core
}//namespace fdapde

#endif //RAND_RANGE_FINDER_H