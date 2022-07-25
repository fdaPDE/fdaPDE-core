#include "FSPAI.h"

namespace fdaPDE{
namespace core{
namespace NLA{

  // build system matrix A(p1, p2) given sparsity patterns p1 and p2. The result is a |p1| x |p2| dense matrix
  void FSPAI::extractSystem(const column_sparsity_pattern& p1,
			    const column_sparsity_pattern& p2,
			    const Eigen::Index& k) {
    // resize memory buffers
    Ak_.resize(p1.size(), p2.size());
    bk_.resize(p2.size(), 1);
  
    // (*it1.first, *it2.first) is each time a pair of coordinates in the cross product p1 X p2
    for(auto it1 = std::make_pair(p1.begin(), 0);
	it1.first != p1.end();
	it1.first++, it1.second++){
      for(auto it2 = std::make_pair(p2.begin(), 0);
	  it2.first != p2.end();
	  it2.first++, it2.second++){
      
	Ak_(it1.second, it2.second) = A_.coeff(*it1.first, *it2.first);
      }
      bk_[it1.second] = A_.coeff(*it1.first, k);
    }
    return;
  }

  // update approximate inverse of column k
  void FSPAI::updateApproximateInverse(const Eigen::Index& k, const DVector<double>& bk, const DVector<double>& yk,
				       const column_sparsity_pattern& tildeJk){
    // compute diagonal entry l_kk
    double l_kk = 1.0/(std::sqrt(A_.coeff(k,k) - bk.transpose().dot(yk)));
    Lk_[k] = l_kk;

    // update other entries according to current sparsity pattern tildeJk
    for(auto it = std::make_pair(tildeJk.begin(), 0);
	it.first != tildeJk.end();
	++it.first, ++it.second){

      Lk_[*it.first] = -l_kk*yk[it.second];
    }
    return;
  }

  // select candidate indexes for sparsity pattern update for column k (this reflects in a modification to the hatJk_ structure)
  void FSPAI::selectCandidates(const Eigen::Index& k) {
    // computation of candidate rows to enter in the sparsity pattern of column Lk_
    for(auto row = deltaPattern_.begin(); row != deltaPattern_.end(); ++row){
      for(auto j = sparsityPattern_.at(*row).begin(); j != sparsityPattern_.at(*row).end(); ++j)
	if(*j > k){
	  candidateSet_.insert(*j);
	}
    }
    // clear delta pattern, waiting for a new one...
    deltaPattern_.clear();
    hatJk_.clear();
    
    // for efficiency reasons, compute the value of A(j, Jk)^T * Lk[Jk] here once and store for later use
    for(Eigen::Index j : candidateSet_){
      if(J_[k].find(j) == J_[k].end()){
	double v = 0;
	// compute A(j, jK)*Lk(jK)
	for(auto it = J_[k].begin(); it != J_[k].end(); ++it){
	  v += A_.coeff(j, *it)*Lk_[*it]; 
	}
	// index j is a possible candidate for updating the sparsity pattern of the solution along column k.
	hatJk_.emplace(j, v);
      }
    }
    return;
  }

  // constructor
  FSPAI::FSPAI(const Eigen::SparseMatrix<double>& A) : A_(A), n_(A.rows()) {
    // initialize the sparsity pattern to the identity matrix
    J_.resize(n_);
    for(std::size_t i = 0; i < n_; ++i){
      J_[i].insert(i);
    }

    // pre-allocate memory
    L_.resize(n_, n_);
    Lk_.resize(n_, 1);

    // compute sparsity pattern of matrix A once, cache for reuse
    for (int k=0; k<A_.outerSize(); ++k){
      for (Eigen::SparseMatrix<double>::InnerIterator it(A_, k); it; ++it){
	sparsityPattern_[it.row()].emplace(it.col());
      }
    }
  }

  // compute the Factorize Sparse Approximate Inverse of A using a K-condition number minimization method
  // alpha:   number of sparsity pattern updates to compute for each column k of A_
  // beta:    number of indexes to augment the sparsity pattern of Lk_ per update step
  // epsilon: do not consider an entry of A_ as valid if it causes a reduction to its K-condition number lower than epsilon
  void FSPAI::compute(unsigned alpha, unsigned beta, double epsilon){
    
    // eigen requires a list of triplet to construct a sparse matrix in an efficient way
    std::list<Eigen::Triplet<double>> tripetList;

    // cycle over each column of the sparse matrix A_
    for(std::size_t k = 0; k < n_; ++k){
      Lk_.fill(0); // reset column vector Lk_
      candidateSet_.clear();    // reset candidateSet
      deltaPattern_.insert(k);  // init deltaPattern to allow for sparsity pattern updates
      
      // perform alpha steps of approximate inverse update along column k
      for(std::size_t s = 0; s < alpha; ++s){
	// if sparsity pattern has reached convergence given the supplied epsilon, the approximate inverse along this
	// column cannot change. As a result the approximate inverse cannot change and any further computation can be skipped.
	if(!deltaPattern_.empty()){
	  column_sparsity_pattern tildeJk = J_[k]; // extract sparsity pattern
	  tildeJk.erase(k); 
	
	  if(tildeJk.empty()){
	    // skip computation if tildeJk is empty. No linear system to solve here, just compute the diagonal
	    // element of the current approximate inverse
	    double l_kk = 1.0/(std::sqrt(A_.coeff(k, k)));
	    Lk_[k] = l_kk;
	  }else{
	    // we must find the best vector fixed its sparsity pattern minimizing the K-condition number of L^T*A*L.
	    // It can be proven that this problem is equivalent to the solution of a small dense SPD linear system
	    //         A(tildeJk, tildeJk)*yk = Ak(tildeJk)

	    // define system matrix:        A(tildeJk, tildeJk)
	    // define rhs of linear system: Ak(tildeJk)
	    extractSystem(tildeJk, tildeJk, k);
	  
	    // solve linear system A(tildeJk, tildeJk)*yk = Ak(tildeJk) using Cholesky factorization (system is SPD)
	    choleskySolver_.compute(Ak_);
	    DVector<double> yk = choleskySolver_.solve(bk_); // compute yk = A(tildeJk, tildeJk)^{-1}*Ak(tildeJk)
	  
	    // update approximate inverse
	    updateApproximateInverse(k, bk_, yk, tildeJk);	  
	  }
	  
	  // search for best update of the sparsity pattern
	  // computation of candidate indexes for sparsity pattern update: hatJk_ contains (candidateID j, value of A(j, Jk)^T * Lk[Jk])
	  selectCandidates(k);
	  
	  // use average value heuristic for selection of best candidates
	  double tau_k = 0;   // the average improvement to the K-condition number
	  double max_tau = 0; // the maximum possible improvement.
	  
	  for(auto it = hatJk_.begin(); it != hatJk_.end(); ++it){
	    // compute tau_jk value
	    double tau_jk = std::pow(it->second, 2)/A_.coeff(it->first, it->first);
	    // can overwrite stored data (not required anymore after computation of tau_jk)
	    hatJk_[it->first] = tau_jk;
	    
	    // update average value
	    tau_k += tau_jk;
	    // update maximum value
	    if(tau_jk > max_tau) max_tau = tau_jk;
	  }

	  // if the best improvement is higher than accetable treshold
	  if(max_tau > epsilon){
	    tau_k /= hatJk_.size();

	    // select most promising first beta entries according to average heuristic
	    for(std::size_t idx = 0; idx < beta && !hatJk_.empty(); ++idx){
	      // find maximum
	      auto max = hatJk_.begin();
	      for(auto it = hatJk_.begin(); it != hatJk_.end(); ++it){
		if(it->second >= max->second) max = it; 
	      }
	      // sparsity pattern update
	      if(max->second > tau_k){
		J_[k].insert(max->first);
		deltaPattern_.insert(max->first); 
	      }
	      // erase current maximum to start a new maximum search until idx == beta
	      hatJk_.erase(max);
	    }
	  }
	}
      }
      // save approximate inverse of column k
      for(std::size_t i = 0; i < n_; ++i){
	if(Lk_[i] != 0) tripetList.push_back(Eigen::Triplet<double>(i, k, Lk_[i]));
      }	        
    }
    // build final result
    L_.setFromTriplets(tripetList.begin(), tripetList.end());
    return;
  }

}}}
