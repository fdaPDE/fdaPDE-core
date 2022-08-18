#ifndef __SPARSE_BLOCK_MATRIX_H__
#define __SPARSE_BLOCK_MATRIX_H__

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cmath>
#include <cstddef>

namespace fdaPDE{
namespace core{
namespace NLA{

  template <typename T, unsigned int M, unsigned int N>
  class SparseBlockMatrix {
  private:
    Eigen::SparseMatrix<T> m_;
    Eigen::Index cols_, rows_;
  public:
    // initialize from list of matrices
    template <typename... E>
    SparseBlockMatrix(const E&... m) {
      // check correct number of blocks is passed as parameter
      static_assert(sizeof...(E) == M*N);

      std::vector<unsigned> blockCols{}, blockRows{};
      blockCols.resize(M*N), blockRows.resize(M*N);
      
      // use C++17 fold expression to loop over the parameter pack and extract the overall size of the new matrix
      std::size_t ctr = 0;
      ([&] {
	if(std::floor(ctr/M) == 0) cols_ += m.cols();
	if(ctr%M == 0) rows_ += m.rows();
	// store row-column information for each block
	blockRows[ctr] = m.rows();
	blockCols[ctr] = m.cols();
	ctr++;
      }(), ...);

      // build new matrix
      m_.resize(rows_, cols_);
      std::list<Eigen::Triplet<T>> tripletList;
      ctr = 0;
      std::size_t i,j = 0;
      // unfold argument list via fold expression
      ([&] {
	// cast block to SparseMatrix
	Eigen::SparseMatrix<T> block = m;
	// define row/column offset for block
	if (std::floor(ctr/M) == 0) i = 0;
	else i = blockRows[ctr - M];
	if (ctr%M == 0) j = 0;
	else j += blockCols[ctr - 1];

	// iterate over nonzero elements and push back in triplet list
	for (std::size_t k = 0; k < block.outerSize(); ++k){
	  for (Eigen::SparseMatrix<double>::InnerIterator it(block,k); it; ++it){
	    tripletList.push_back(Eigen::Triplet<T>(i + it.row(), j + it.col(), it.value()));
	  }
	}
	ctr++;
      }(), ...);
      
      m_.setFromTriplets(tripletList.begin(), tripletList.end());
      m_.makeCompressed();
    };
    
    Eigen::SparseMatrix<T>& derived() { return m_; }
  };  
  
}}}
#endif // __SPARSE_BLOCK_MATRIX_H__
