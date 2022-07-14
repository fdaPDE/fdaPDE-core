#include "../utils/Symbols.h"

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/ConfigureVectorization.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/SparseCholesky/SimplicialCholesky.h>
#include <Eigen/src/SparseCore/SparseMatrix.h>
#include <Eigen/src/SparseCore/SparseUtil.h>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <Eigen/IterativeLinearSolvers>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>
#include <utility>
#include <Eigen/SparseQR>

#include <variant>

#include "SMW.h"
#include "FSPAI.h"
#include <random>
#include <chrono>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseCholesky>

Eigen::SparseMatrix<double> generateRandomSparseMatrix(std::size_t rows,
                                                       std::size_t cols,
						       std::size_t fill_in) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0,1.0);
  std::uniform_int_distribution<int> distI(0, cols-1);
  
  std::vector<Eigen::Triplet<double> > tripletList;
  for(int i = 0; i < rows; ++i){
    double v_ii = dist(gen);
    tripletList.push_back(Eigen::Triplet<double>(i, i, v_ii));
    for(int j = 0; j < fill_in; ++j){
      double v_ic = dist(gen);
      int c = distI(gen);
      if(c != i)
	tripletList.push_back(Eigen::Triplet<double>(i, c, v_ic));
    }
  }
  Eigen::SparseMatrix<double > result(rows,cols);
  result.setFromTriplets(tripletList.begin(), tripletList.end());   //create the matrix
  
  return result;
}

void displaySparsistyPattern(const Eigen::SparseMatrix<double> &m) {
  std::cout << Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>(m.cast<bool>()) << std::endl;
  return;
}

int main(){

  /*
  // woodbury test
  static constexpr unsigned N = 70;
  static constexpr unsigned q = 2;
  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> AA = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Random(N,q);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> BB = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Random(q,N);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> HH = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Random(q,q);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A = AA*HH*BB;

  //std::cout << A << std::endl;
  
  Eigen::MatrixXd R1 = Eigen::MatrixXd(generateRandomSparseMatrix(N, N));
  Eigen::MatrixXd R0 = Eigen::MatrixXd(generateRandomSparseMatrix(N, N));

  //std::cout << R1 << std::endl;
  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M;
  M.resize(2*N, 2*N);
  M << A, R1, R1.transpose(), R0;

  //  std::cout << "system matrix" << std::endl;
  //  std::cout << M << std::endl;

  // back to sparse representation
  Eigen::SparseMatrix<double> SpM = M.sparseView();
  Eigen::VectorXd b = Eigen::VectorXd::Ones(2*N);
  
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  
  // auto t1 = high_resolution_clock::now();
  
  // solve with SparseLU solver of eigen
  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > SparseLU;
  SparseLU.analyzePattern(SpM);    // Compute the ordering permutation vector from the structural pattern of A
  SparseLU.factorize(SpM);         // compute LU factorization of matrix A
  
  Eigen::VectorXd sol = SparseLU.solve(b);

  // auto t2 = high_resolution_clock::now();

  // //std::cout << "system solution" << std::endl;
  // //std::cout << sol << std::endl;

  // duration<double, std::milli> ms_double = t2 - t1;

  // std::cout << "standard sparseQR solver on full matrix: " << ms_double.count() << "ms\n";
  
  
  std::cout << "woodbury solution" << std::endl;
  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(N,N);
  B.setIdentity();

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Z = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(N,q);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> U;
  U.resize(2*N, q);
  U << AA, Z;

  //  std::cout << U << std::endl;
  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V;
  V.resize(q, 2*N);
  V << BB, Z.transpose();
  
  DMatrix invHH = HH.inverse();
  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M1;
  M1.resize(2*N, 2*N);
  M1 << Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(N,N), R1, R1.transpose(), R0;
  
  //  std::cout << M1 + (U*A*V) << std::endl;
  
  SpMatrix spM1 = M1.sparseView();
  SpMatrix spU = U.sparseView();
  SpMatrix spV = V.sparseView();

  
  auto t3 = high_resolution_clock::now();
    
  DVector w_sol = SMW::solve(spM1, spU, invHH, spV, b);

  auto t4 = high_resolution_clock::now();
  duration<double, std::milli> ms_double2 = t4 - t3;

  std::cout << "SMW solver: " << ms_double2.count() << "ms\n";  

  
  //  std::cout << w_sol << std::endl;


  
  std::cout << "distance between solutions" << std::endl;
  std::cout << (sol - w_sol).norm() << std::endl;

  */

  Eigen::SparseMatrix<double> ex33Mat;
  Eigen::loadMarket(ex33Mat, "matrix_data/ex33/ex33.mtx");

  // this is a symmetrix matrix, matrix market format only stores the lower triangular part. Symmetrize the matrix here
  Eigen::SparseMatrix<double> tmp = ex33Mat.transpose().triangularView<Eigen::StrictlyUpper>();
  ex33Mat = ex33Mat + tmp;

  //  displaySparsistyPattern(ex33Mat);
  
  std::cout << "load matrix from matrix market" << std::endl;
  std::cout << "   number of rows:    " << ex33Mat.rows() << std::endl;
  std::cout << "   number of columns: " << ex33Mat.cols() << std::endl;
  
  FSPAI fspai(ex33Mat);

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();
  fspai.compute(150, 3, 0.005);
  auto t2 = high_resolution_clock::now();

  /* Getting number of milliseconds as an integer. */
  auto ms_int = duration_cast<milliseconds>(t2 - t1);

  Eigen::SparseMatrix<double> inv = fspai.getL();
  
  //  std::cout << inv << std::endl;
  std::cout << "inverse computed" << std::endl;
  std::cout << "   number of rows:    " << inv.rows() << std::endl;
  std::cout << "   number of columns: " << inv.cols() << std::endl;
  std::cout << "   time:              " << ms_int.count() << "ms" << std::endl;

  std::cout << "number of non-zero elements ex33: " << ex33Mat.nonZeros() << std::endl;
  std::cout << "number of non-zero elements: " << inv.nonZeros() << std::endl;
  std::cout << "inverse sparsity pattern: " << std::endl;
  //  displaySparsistyPattern(inv);

  std::cout << "compute cholesky factorization of original matrix: " << std::endl;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > choleskySolver;
  choleskySolver.analyzePattern(ex33Mat);
  choleskySolver.factorize(ex33Mat);

  Eigen::SparseMatrix<double> L;
  L = choleskySolver.matrixL();

  Eigen::SparseMatrix<double> I;
  I.resize(ex33Mat.rows(), ex33Mat.cols());
  I.setIdentity();

  std::cout << "Frobenius norm L - I:         " << (L- I).norm() << std::endl;
  std::cout << "Frobenius norm approximation: " << (L*inv - I).norm() << std::endl;
  
  // std::cout << "########################" << std::endl;
  // std::cout << Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(ex33Mat).block<10,10>(0,0) << std::endl;

  // std::cout << "########################" << std::endl;
  // std::cout << Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(inv).col(0) << std::endl;

  Eigen::SparseMatrix<double> prec_fspai;
  //Eigen::loadMarket(prec_fspai, "matrix_data/msc01440/precond.mtx");
  Eigen::loadMarket(prec_fspai, "../../../fspai-1.1/bin/precond.mtx");
  
  std::cout << "########################" << std::endl;
  std::cout << Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(inv).block<10,10>(0,0) << std::endl;

  std::cout << "########################" << std::endl;
  std::cout << Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(prec_fspai).block<10,10>(0,0) << std::endl;

  std::cout << "Frobenius norm fspai: " << (L*prec_fspai - I).norm() << std::endl;
  
  return 0;
}
