//#define EIGEN_MATRIX_PLUGIN "../utils/MatrixAddons.h"

#include "../utils/Symbols.h"

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <Eigen/IterativeLinearSolvers>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../utils/fields/VectorField.h"
#include "../utils/fields/VectorFieldExpressions.h"

#include "../utils/fields/DotProduct.h"
#include "operators/BilinearFormExpressions.h"
using fdaPDE::core::DotProduct;

#include "../../../matplotlibcpp.h"

#include "../utils/CompileTime.h"
//#include "FiniteElement.h"
#include "basis/LagrangianBasis.h"
#include "basis/MultivariatePolynomial.h"
#include "Assembler.h"
#include "../MESH/engines/AlternatingDigitalTree/ADT.h"
#include "../MESH/engines/BruteForce/BruteForce.h"
#include "../MESH/Mesh.h"
#include "integration/Integrator.h"

//namespace plt = matplotlibcpp;
#include "../utils/fields/ScalarField.h"
#include "../utils/fields/ScalarFieldExpressions.h"
#include "operators/Gradient.h"

using fdaPDE::core::ScalarField;
using fdaPDE::core::MESH::ADT;
using fdaPDE::core::MESH::Mesh;

#include "operators/Laplacian.h"
#include "operators/Identity.h"
#include "PDE.h"
#include "../MESH/CSVReader.h"
#include "basis/SplineBasis.h"

#include "operators/BilinearFormTraits.h"
#include "solvers/FEMStandardSpaceSolver.h"
#include "solvers/FEMStandardSpaceTimeSolver.h"

int main(){

  //const unsigned N = 3;
  //const unsigned R = 2;

  // constexpr std::array<unsigned, ct_binomial_coefficient(R+N, R)*N> prova = Polynomial<N,R>::expVect;
  
  // std::cout << "ci sono" << std::endl;

  // for(size_t i = 0; i < ct_binomial_coefficient(R+N, R); ++i){

  //   for(size_t j = 0; j < N; ++j){
  //     std::cout << prova[i*N + j];
  //   }
  //   std::cout << std::endl;
  // }

  // create a plane on 2D space

  // std::array<double, 6> coeff = {1,-1,-1, 1, 1, 1};
  
  // MultivariatePolynomial<2,2> plane = MultivariatePolynomial<2,2>(coeff);

  //std::cout << plane.gradient()(SVector<2>(0.25,0.25)) << std::endl;

  // build a lagrange basis over a 2D unit simplex
  // std::array<std::array<double, 2>, 6> nodes;
  // nodes[0] = {0,0};
  // nodes[1] = {1,0};
  // nodes[2] = {0,1};
  // nodes[3] = {0.5,0};
  // nodes[4] = {0,0.5};
  // nodes[5] = {0.5,0.5};
  
  
  //LagrangianBasis<2, 2> basis(nodes);
  
  //std::cout << basis.getBasisElement(2)(SVector<2>(0,1)) << std::endl;

  // ReferenceBasis<2, 2> r;
  
  // // try first plot
  // double step = 0.05;
  // std::vector<std::vector<double>> ev, x, y;
  
  // unsigned int l = 1/step;
  // for(size_t i = 0; i<=l; i++){
  //   std::vector<double> ev_row, x_row, y_row;
  //   for(size_t j = 0; j<=l; j++){
  //     if(j<=l-i){
  // 	x_row.push_back(i*step);
  // 	y_row.push_back(j*step);
  // 	ev_row.push_back((r.getBasis().getBasisElement(3) + r.getBasis().getBasisElement(0))(SVector<2>(i*step, j*step)));
  //     }else{
  // 	x_row.push_back(std::numeric_limits<double>::quiet_NaN());
  // 	y_row.push_back(std::numeric_limits<double>::quiet_NaN());
  // 	ev_row.push_back(std::numeric_limits<double>::quiet_NaN());
  //     }
  //   }
  //   x.push_back(x_row);
  //   y.push_back(y_row);
  //   ev.push_back(ev_row);    
  // }
  
  // matplotlibcpp::plot_surface(x,y,ev);
  // matplotlibcpp::show();  

  // MultivariatePolynomial<2,2> plane1 = MultivariatePolynomial<2,2>(coeff);
  // MultivariatePolynomial<2,2> plane2 = MultivariatePolynomial<2,2>(coeff);

  // std::cout << plane1.gradient()(SVector<2>(1,1)) << std::endl;
  // std::cout << "--------------" << std::endl;

  // std::cout << plane2.gradient()(SVector<2>(1,1)) << std::endl;
  // std::cout << "--------------" << std::endl;
  
  // // callable scalar product
  // auto sp = plane1.gradient().dot(plane2.gradient());
  // std::cout << sp(SVector<2>(1,1), SVector<2>(1,1)) << std::endl;
  
  // //  auto e = plane + 1;
  // auto e = plane + 1 + 2*plane;
  // double x = e(SVector<2>(0.25,0.25));
  // std::cout << x << std::endl;
 
  // std::cout << "vero poly valore: " << plane(SVector<2>(0.25, 0.25)) << std::endl;
  // //std::cout << (basis.getBasisElement(2)*basis.getBasisElement(1))(SVector<2>(0.25,0.25)) << std::endl;

  // // try to solve a 2D poisson problem with homogeneous boundary conditions
  // Mesh<2,2> m("../MESH/points.csv", "../MESH/edges.csv", "../MESH/triangles.csv", "../MESH/neighbors.csv");
  // std::cout << "mesh info loaded..." << std::endl;

  // Assembler<2,2,1> ass;
  // auto A = ass.assemble(m);
  
  // std::function<double(SVector<2>)> forcing = [](SVector<2> x) -> double { return 4; };
  // auto b = ass.forcingTerm(m, forcing);

  //  std::cout << A << std::endl;
  
   //Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
   //solver.analyzePattern(A);
   //solver.factorize(A);

   //std::cout << "..................." << std::endl;
   //std::cout << solver.solve(b) << std::endl;

  //  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper > solver;

  //Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > solver;
  // A.makeCompressed();
  // Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
  // solver.compute(A);
  
  // // if(solver.info()!=Eigen::Success) std::cout << "FAILURE" << std::endl;
  // // std::cout << solver.info() << std::endl;
  
  // Eigen::VectorXd sol = solver.solve(b);
  // //  std::cout << sol << std::endl;

  // // evaluate solution
  // std::vector<SVector<2>> listPoint;  
  // ADT<2,2> tree(m);

  // std::vector<double> solVect{};

  // std::vector<std::vector<double>> ev, xC, yC; // plot vectors

  // for(double x = m.getMeshRange()[0].first; x < m.getMeshRange()[0].second; x+=0.1){
  //   std::vector<double> x_row, y_row;
  //   for(double y = m.getMeshRange()[1].first; y < m.getMeshRange()[1].second; y+=0.1){
  //     listPoint.push_back(SVector<2>(x,y));

  //     x_row.push_back(x); // plot
  //     y_row.push_back(y);
  //   }

  //   xC.push_back(x_row); // plot
  //   yC.push_back(y_row);
  // }

  // int jj = 0; // plot
  // std::vector<double> ev_row;
  // for(const SVector<2>& point : listPoint){
  //   // search element containing point
  //   std::shared_ptr<Element<2,2>> elem = tree.search(point);

  //   double solValue = 0;
  //   if(elem != nullptr){
  //     // build a lagrangian basis over the element
  //     std::array<std::array<double, 2>, 3> nodes;
  //     std::array<unsigned int, 3> nodeIDxs;
  //     for(size_t j = 0; j < 3; ++j){ // 3 deve scalare sul numero di nodi
  // 	nodes[j] = {elem->getFESupport()[j].second[0], elem->getFESupport()[j].second[1]};
  // 	nodeIDxs[j] = elem->getFESupport()[j].first;
  //     }
  //     LagrangianBasis<2,1> basis(nodes);

  //     // evaluate the solution
  //     for(size_t j = 0; j < 3; ++j){
  // 	solValue += sol[nodeIDxs[j]] * basis.getBasisElement(j)(point);
  //     }
  //     solVect.push_back(solValue);
  //   }else{
  //     solVect.push_back(std::numeric_limits<double>::quiet_NaN());
  //   }

  //   if(jj < xC[0].size()){
  //     ev_row.push_back(solValue);
  //     ++jj;
  //   }else{
  //     ev.push_back(ev_row);
  //     ev_row.erase(ev_row.begin(), ev_row.end());
  //     jj = 0;
  //     ev_row.push_back(solValue);
  //     ++jj;
  //   }
  // }
  // ev.push_back(ev_row);
  
  // std::cout << listPoint.size() << std::endl;
  // std::cout << xC[0].size() << std::endl;
  // std::cout << yC[0].size() << std::endl;
  // std::cout << ev[0].size() << std::endl;
  // std::cout << xC.size() << std::endl;
  // std::cout << yC.size() << std::endl;
  // std::cout << ev.size() << std::endl;
  
  // //for(double z : solVect)  std::cout << z << std::endl;

  // // double step = 0.1;
  
  // // unsigned int l = (m.getMeshRange()[0].second - m.getMeshRange()[0].first)/step;
  // // for(size_t i = 0; i<=l; i++){
  // //   std::vector<double> ev_row, x_row, y_row;
  // //   for(size_t j = 0; j<=l; j++){
  // //     x_row.push_back(listPoint[i+j][0]);
  // //     std::cout << listPoint[i+j][0];
  // //     y_row.push_back(listPoint[i+j][1]);
  // //     std::cout << listPoint[i+j][1];
  // //     ev_row.push_back(solVect[i+j]);
  // // 	//std::cout << solVect[i+j] << std::endl;
  // //   }
  // //   xC.push_back(x_row);
  // //   yC.push_back(y_row);
  // //   ev.push_back(ev_row);    
  // // }

  // matplotlibcpp::plot_surface(xC,yC,ev);
  // matplotlibcpp::show();  
  /*
  std::cout << "plane evaluation at point: " << plane(SVector<2>(1,1)) << std::endl;
  std::cout << "plane gradient evaluation at point: \n" << plane.gradient()(SVector<2>(1,1)) << std::endl;
  std::cout << "try matrix-vector product: \n";

  SMatrix<2> J { {1,1}, {1,1} };

  VectorField<2> field = plane.gradient();

  auto k = J*field;
  
  std::cout << (J*field)(SVector<2>(1,1)) << std::endl;

  SVector<2> x { 1 ,1 };

  auto l = J*field + field;
  
  std::cout << (J*field + field - field)(SVector<2>(1,1)) << std::endl;

  // arithmetic over scalar fields

  std::function<double(SVector<2>)> f = [](SVector<2> x) -> double{
    return std::pow(x[0], 2) + std::pow(x[1], 2) + 1;
  };
  ScalarField<2> field1(f);
  
  std::function<double(SVector<2>)> g = [](SVector<2> x) -> double{
    return std::pow(x[0], 2) + std::pow(x[1], 2) + 1;
  };
  ScalarField<2> field2(g);
  
  auto gg = 3*field1 + plane;

  std::cout << gg(SVector<2>(1,1)) << std::endl;
  */
  
  // try to solve a 2D poisson problem with homogeneous boundary conditions
  // Mesh<2,2> m("circle_points.csv", "circle_edges.csv", "circle_triangles.csv", "circle_neighbors.csv", "circle_boundary_markers.csv");
  // std::cout << "mesh info loaded..." << std::endl;

  /*  Integrator<2,6> integrator;
  Element<2,2> e = *m.requestElementById(0);
  ReferenceBasis<2, 2, 1> fe;                    // reference element where functional information is defined

  auto b = fe.getBasis().getBasisElement(0);
  
  double v = integrator.integrate(e, b);

  double vmano = 0;
  for(size_t iq = 0; iq < integrator.getTable().num_nodes; ++iq){
    vmano += b(SVector<2>(integrator.getTable().nodes[iq].data()))*integrator.getTable().weights[iq];
    //std::cout << "valore a mano: " << b(SVector<2>(integrator.getTable().nodes[iq].data())) << std::endl;
  }
  
  vmano *= std::abs(e.getBaryMatrix().determinant())/ct_factorial(2);
  //std::cout << "cons mano:" << std::abs(e.getBaryMatrix().determinant())/ct_factorial(2) << std::endl;
  
  std::cout << "integrator basis function 0 su elemento e: " << v << std::endl;
  std::cout << "integrazione manuale basis function 0 su elemento e: " << vmano << std::endl;
  
  // create a fake element for the unit triangle
  std::array<std::pair<unsigned, SVector<2>>, 3> FEsupport_ = {std::pair<unsigned, SVector<2>>(0, SVector<2>(0,0)),
							       std::pair<unsigned, SVector<2>>(2, SVector<2>(1,0)),
  							       std::pair<unsigned, SVector<2>>(1, SVector<2>(0,1))};
  std::array<SVector<2>, 3> coords_ = {SVector<2>(0,0), SVector<2>(0,1), SVector<2>(1,0)};
  std::array<unsigned, 3> neigh = {1,1,1};
  std::array<std::pair<unsigned, unsigned>, 3> boundary = {std::pair<unsigned, unsigned>(0,1),
							   std::pair<unsigned, unsigned>(1,1),
							   std::pair<unsigned, unsigned>(2,1)};
  Element<2,2> fake = Element<2,2>(0, FEsupport_, coords_, neigh, boundary);

  std::function<double(SVector<2>)> constant = [](SVector<2> x) -> double {return 1;};
  
  v = integrator.integrate(fake, b);
  std::cout << "integrator on reference triangle: " << v << std::endl;

  std::cout << "INTEGRAZIONE TESTATA E CORRETTA FINO A QUI" << std::endl;
  
  // check if integral of gradient is correct
  std::function<double(SVector<2>)> f1 = [](SVector<2> x) -> double { return 1; };
  std::function<double(SVector<2>)> f2 = [](SVector<2> x) -> double { return 1; };

  std::array<std::function<double(SVector<2>)>, 2> arr = {f1, f2};
  VectorField<2> vfield(arr);
  
  auto ip = vfield.dot(vfield);

  std::cout << "valore inner product " << ip(SVector<2>(0,0)) << std::endl;
  std::cout << "CORRETTO" << std::endl;

  std::cout << "\nbarycentric matrix" << std::endl;
  std::cout << e.getBaryMatrix() << std::endl;

  std::cout << "-----" << std::endl;
  std::cout << "(1,0) from reference to current\n" << e.getBaryMatrix()*SVector<2>(1,0) + e.getCoords()[0] << std::endl;       //from reference element to current element
  std::cout << "from current to reference (1,0)\n" << e.getInvBaryMatrix()*(e.getCoords()[1] - e.getCoords()[0]) << std::endl; //from current element to reference element
  std::cout << "----- CORRETTO" << std::endl;
  
  std::cout << "transpose of the inverse barycentric matrix" << std::endl;
  Eigen::Matrix<double, 2, 2> invJ = e.getInvBaryMatrix().transpose();

  std::cout << invJ << std::endl;

  std::cout << "gradient over element e" << std::endl;
  std::cout << invJ * ip.lhs_(SVector<2>(0,0)) << std::endl;
  std::cout << "TESTATO A MANO E CORRETTO" << std::endl;

  std::cout << "risultato atteso: 2*area_of_element: 1" << std::endl;
  std::cout << "integrator of inner product on reference triangle: " << integrator.integrate(fake, ip) << std::endl;

  std::cout << "------ CORRETTO" << std::endl;
  
  std::cout << "area of element: " << std::abs(e.getBaryMatrix().determinant())/2 << std::endl;
  std::cout << "integral of constant 1 on element e: " << integrator.integrate(e, constant) << std::endl;
  std::cout << "------ CORRETTO" << std::endl;
 
  
  std::cout << "gradient of basis elements:" << std::endl;
  std::cout << fe.getBasis().getBasisElement(0).gradient()(SVector<2>(0,0)) << std::endl;
  
  ip = fe.getBasis().getBasisElement(0).gradient().dot(fe.getBasis().getBasisElement(0).gradient());
  std::cout << "valore inner product phi_0.dot(phi_0): " << ip(SVector<2>(0,0)) << std::endl;  

  auto ip2 = vfield.dot(vfield);

  // build a lagrangian basis over the element
  std::array<std::array<double, 2>, 3> nodess;
  for(size_t j = 0; j < 3; ++j){ // 3 deve scalare sul numero di nodi
    nodess[j] = {e.getFESupport()[j].second[0], e.getFESupport()[j].second[1]};
    // std::cout << e.getFESupport()[j].second[0] << " - " << e.getFESupport()[j].second[1] << std::endl;
  }
  LagrangianBasis<2,1> basis(nodess);
  
  std::cout << "integrator of inner product on generic triangle: \n valore atteso: "<<
    basis.getBasisElement(0).gradient().dot(basis.getBasisElement(0).gradient())(SVector<2>(0,0))*std::abs(e.getBaryMatrix().determinant())/2
	    << "\nrisultato: " << integrator.integrate(e, ip2) << std::endl;

  std::cout << "CORRETTO" << std::endl;
  
  std::cout << "coords element: \n";
  std::cout << e.getCoords()[0] << std::endl;
  std::cout << "----" << std::endl;
  std::cout << e.getCoords()[1] << std::endl;
  std::cout << "----" << std::endl;
  std::cout << e.getCoords()[2] << std::endl;


  std::cout << "gradient of basis function over element" << std::endl;
  std::cout << basis.getBasisElement(0).gradient()(SVector<2>(0,0)) << std::endl;
  std::cout << "----" << std::endl;
  std::cout << basis.getBasisElement(1).gradient()(SVector<2>(0,0)) << std::endl;
  std::cout << "----" << std::endl;
  std::cout << basis.getBasisElement(2).gradient()(SVector<2>(0,0)) << std::endl;
  std::cout << "----" << std::endl;

  std::cout << "evaluation of basis function over element" << std::endl;
  std::cout << basis.getBasisElement(0)(SVector<2>(nodess[0].data())) << std::endl;
  
  std::cout << "FIN QUI CORRETTO" << std::endl;

  std::cout << "integrazione forcing term su elemento e: " << std::endl;
  std::function<double(SVector<2>)> forcing2 = [](SVector<2> x) -> double { return 4; };
  auto fphi = ScalarField<2>(forcing2) * fe.getBasis().getBasisElement(0);
  std::cout << integrator.integrate(e, fphi) << std::endl;
  
  std::cout << "SOLUTION EVALUATION" << std::endl;*/
  // double step = 0.001;
  // std::vector<std::vector<double>> ev, xx, yy;
  
  // unsigned int la = 1/step;
  // for(int i = -1000; i<=0; i++){
  //   std::vector<double> ev_row, x_row, y_row;
  //   for(int j = -1000; j<=0; j++){
  //     if(e.contains(SVector<2>(i*step, j*step))){
  // 	x_row.push_back(i*step);
  // 	y_row.push_back(j*step);
  // 	ev_row.push_back(basis.getBasisElement(0)(SVector<2>(i*step, j*step)));
  //     }else{
  // 	x_row.push_back (std::numeric_limits<double>::quiet_NaN());
  // 	y_row.push_back (std::numeric_limits<double>::quiet_NaN());
  // 	ev_row.push_back(std::numeric_limits<double>::quiet_NaN());
  //     }
  //   }
  //   xx.push_back(x_row );
  //   yy.push_back(y_row );
  //   ev.push_back(ev_row);    
  // }
  
  // matplotlibcpp::plot_surface(xx,yy,ev);
  // matplotlibcpp::show();


  
  
  //auto form = Laplacian() - 1000*Identity();
  
  //auto ff = DotProduct(SVector<2>(1,1), Gradient());
  
  
  // Assembler<2,2,1> ass(m);
  //auto A = ass.assemble(form);

  //std::cout << "main :CELLA (212,213): " << A.coeff(212,213) << std::endl;
  //std::cout << "main :CELLA (213,212): " << A.coeff(213,212) << std::endl;
  //std::cout << "main :CELLA (211,212): " << A.coeff(212,212) << std::endl;

  /*
  
  Mesh<2,2> m("circle_points.csv", "circle_edges.csv", "circle_triangles.csv", "circle_neighbors.csv", "circle_boundary_markers.csv");
  std::cout << "mesh info loaded..." << std::endl;

  CSVReader reader;
  CSVFile<double> boundary_data = reader.parseFile<double>("circle_data.csv");
  Eigen::Matrix<double, Eigen::Dynamic, 1> boundaryMatrix = boundary_data.toEigen();

  SMatrix<2> K { {10,0}, {0,5} };
  
  auto form = Laplacian() + dot(SVector<2>(2,1), Gradient()) - 100*Identity();
  
  std::function<double(SVector<2>)> forcing = [](SVector<2> x) -> double { return 4; };
  ScalarField<2> forcing_field(forcing);
  
  DVector forcingData = forcing_field.discretize(m);

  PDE problem(m, form, forcingData);
  problem.setDirichletBC(boundaryMatrix);

  //  std::cout << problem.getForcingData();
  
  LagrangianBasis<2, 1> basis{};  // define functional basis
  Integrator<2, 6> integrator{};  // define integrator
  
  // define solver
  FEMStandardSpaceSolver solver(basis, integrator);
  solver.solve(problem);

  DVector sol = solver.getSolution();

  // for(const auto& i : problem.getBoundaryData()){
  //   std::cout << i.first << "-" << i.second << std::endl;
  //   std::cout << "............." << std::endl;
  // }

  

  /*
  Mesh<2,2> m("circle_points.csv", "circle_edges.csv", "circle_triangles.csv", "circle_neighbors.csv", "circle_boundary_markers.csv");
  std::cout << "mesh info loaded..." << std::endl;

  CSVReader reader;
  CSVFile<double> boundary_data = reader.parseFile<double>("circle_data.csv");
  Eigen::Matrix<double, Eigen::Dynamic, 1> boundaryData = boundary_data.toEigen();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> boundaryMatrix;
  boundaryMatrix.resize(boundaryData.rows(), 10);
  
  SMatrix<2> K { {10,0}, {0,5} };
  
  auto form = dT() + Laplacian();
  
  std::function<double(SVector<2>)> forcing = [](SVector<2> x) -> double { return 0; };
  ScalarField<2> forcing_field(forcing);
  
  DVector forcingData = forcing_field.discretize(m);
  
  PDE problem(m, form, forcingData);
  problem.setDirichletBC(boundaryMatrix);

  LagrangianBasis<2, 1> basis;  // define functional basis
  Integrator<2, 6> integrator;  // define integrator
  
  // define solver
  FEMStandardSpaceSolver solver(basis, integrator);
  solver.solve(problem);

  DVector sol = solver.getSolution();

*/
  
  //std::cout << forcingData << std::endl;
  
  //auto bb = ass.forcingTerm(forcingData);
  //ass.dirichletBoundaryConditions(A, bb, "circle_boundary_markers.csv", "circle_data.csv");

  // impongon omoengea a caso
  // for(size_t j = 0; j < 100; ++j){
  //   // if this node is a boundary node, change the corresponding row in the stiff matrix
  //   // To impose a Dirichlet boundary condition means to introduce an equation of the kind u_j = b_j where j is the index
  //   // of the boundary node and b_j is the boundary value we want to impose on this node. This actually removes one degree
  //   // of freedom from the system. We do so by zeroing out the j-th row of the stiff matrix and set the corresponding
  //   // diagonal element to 1
  //     A.row(j) *= 0;              // zero all entries of this row
  //     A.coeffRef(j,j) = 1;        // set diagonal element to 1 to impose equation u_j = b_j
  //     bb[j] = 0; // impose boundary value
  // }
  
  //std::cout << bb << std::endl;
  
  //A.makeCompressed();
  //Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
  //solver.compute(A);
  
  // if(solver.info()!=Eigen::Success) std::cout << "FAILURE" << std::endl;
  // std::cout << solver.info() << std::endl;
  
  //Eigen::VectorXd sol = solver.solve(bb);

  
  /*
  std::cout << "assemblato, sarà corretto?? :|" << std::endl;
  
  // std::cout << sol << std::endl;
  
  // std::cout << sol << std::endl;
  fdaPDE::core::MESH::ADT<2,2> tree(m);
  
  auto evalSolution = [&tree, &sol](const SVector<2>& point) mutable -> double {
    // search element containing point
    std::shared_ptr<Element<2,2>> elem = tree.search(point);

    double solValue = 0;
    
    if(elem == nullptr) std::cout << "elemento non trovato" << std::endl;
    else{
      std::cout << "elementoID: " << elem->getID() << std::endl;
      // build a lagrangian basis over the element
      std::array<unsigned int, 3> nodeIDxs{};
      for(size_t j = 0; j < 3; ++j){ // 3 deve scalare sul numero di nodi
	nodeIDxs[j] = elem->getFESupport()[j].first;
      }
      LagrangianBasis<2, 1> basis(*elem);

      // evaluate the solution
      for(size_t j = 0; j < 3; ++j){
	solValue += (sol[nodeIDxs[j]]) * basis[j](point);
	std::cout << "node: " << nodeIDxs[j] << std::endl;
	std::cout << "coefficient:  " << sol[nodeIDxs[j]] << std::endl;
	std::cout << "eval of basis: " << basis[j](point) << std::endl;
	std::cout << sol[nodeIDxs[j]] * basis[j](point) << std::endl;
      }
    }
    std::cout << "mo te la printo" << std::endl;
    return solValue;
  };
  
  std::cout << "valutazione soluzione: \n\n" << evalSolution(SVector<2>(0.01,0.01)) << std::endl;

  //  std::cout << sol << std::endl;
  
  // #### SOLUTION EVALUATION
  std::vector<SVector<2>> listPoint;  

  std::vector<double> solVect{};

  std::vector<std::vector<double>> ev, xC, yC; // plot vectors

  for(double x = m.getMeshRange()[0].first - 0.2; x <= m.getMeshRange()[0].second + 0.2; x+=0.05){
    std::vector<double> x_row, y_row;
    for(double y = m.getMeshRange()[1].first - 0.2; y <= m.getMeshRange()[1].second + 0.2; y+=0.05){
      listPoint.push_back(SVector<2>(x,y));

      x_row.push_back(x); // plot
      y_row.push_back(y);

    }
    
    xC.push_back(x_row); // plot
    yC.push_back(y_row);
  }
  
  int jj = 0; // plot
  std::vector<double> ev_row;
  for(const SVector<2>& point : listPoint){
    // search element containing point
    std::shared_ptr<Element<2,2>> elem = tree.search(point);

    double solValue = 0;
    if(elem != nullptr){
      // build a lagrangian basis over the element
      std::array<std::array<double, 2>, 3> nodes{};
      std::array<unsigned int, 3> nodeIDxs{};
      for(size_t j = 0; j < 3; ++j){ // 3 deve scalare sul numero di nodi
	nodes[j] = {elem->getFESupport()[j].second[0], elem->getFESupport()[j].second[1]};
	nodeIDxs[j] = elem->getFESupport()[j].first;
      }
      LagrangianBasis<2,1> basis(nodes);

      // evaluate the solution
      for(size_t j = 0; j < 3; ++j){
	solValue += (sol[nodeIDxs[j]]) * basis[j](point);
      }
      solVect.push_back(solValue);
    }else{
      solVect.push_back(std::numeric_limits<double>::quiet_NaN());
    }

    if(jj < xC[0].size()){
      ev_row.push_back(solValue);
      ++jj;
    }else{
      ev.push_back(ev_row);
      ev_row.erase(ev_row.begin(), ev_row.end());
      jj = 0;
      ev_row.push_back(solValue);
      ++jj;
    }
  }
  ev.push_back(ev_row);

  matplotlibcpp::plot_surface(xC,yC,ev);
  matplotlibcpp::title("Advection-Diffusion-Reaction equation -Laplacian(u) + dot([2,1], Gradient()) - 100*u = 4\n with homogeneous BC on unit circle");
  matplotlibcpp::show();
  // ### END OF EVALUATION
  
  /*


  // proviamo a risolvere un problema di Laplace con condizioni al controrno... uscirà?? :|
  /*
  Assembler<2,2,1> assLaplace(m);
  auto A_Laplace = assLaplace.assemble();

  std::function<double(SVector<2>)> forcing_Laplace = [](SVector<2> x) -> double { return 0; };
  ScalarField<2> forcingLaplace_field(forcing_Laplace);
  auto bbLaplace = assLaplace.forcingTerm(forcingLaplace_field);

  // imponiamo qualche condizione al bordo :)
  assLaplace.dirichletBoundaryConditions(A_Laplace, bbLaplace, "circle_boundary_markers.csv", "circle_data.csv");

  std::cout << "main :CELLA (212,213): " << A_Laplace.coeff(212,213) << std::endl;
  std::cout << "main :CELLA (213,212): " << A_Laplace.coeff(213,212) << std::endl;
  
  A_Laplace.makeCompressed();
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solverLaplace;
  // Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver(A);
  solverLaplace.compute(A_Laplace);
  
  // if(solver.info()!=Eigen::Success) std::cout << "FAILURE" << std::endl;
  // std::cout << solver.info() << std::endl;
  
  Eigen::VectorXd solLaplace = solverLaplace.solve(bbLaplace);
  */
  // #### SOLUTION EVALUATION
  /*  std::vector<SVector<2>> listPoint;  

  std::vector<double> solVect{};

  std::vector<std::vector<double>> ev, xC, yC; // plot vectors

  for(double x = m.getMeshRange()[0].first - 0.2; x <= m.getMeshRange()[0].second + 0.2; x+=0.01){
    std::vector<double> x_row, y_row;
    for(double y = m.getMeshRange()[1].first - 0.2; y <= m.getMeshRange()[1].second + 0.2; y+=0.01){
      listPoint.push_back(SVector<2>(x,y));

      x_row.push_back(x); // plot
      y_row.push_back(y);
    }
    xC.push_back(x_row); // plot
    yC.push_back(y_row);
  }
  
  int jj = 0; // plot
  std::vector<double> ev_row;
  for(const SVector<2>& point : listPoint){
    // search element containing point
    std::shared_ptr<Element<2,2>> elem = tree.search(point);

    double solValue = 0;
    if(elem != nullptr){
      // build a lagrangian basis over the element
      std::array<std::array<double, 2>, 3> nodes{};
      std::array<unsigned int, 3> nodeIDxs{};
      for(size_t j = 0; j < 3; ++j){ // 3 deve scalare sul numero di nodi
	nodes[j] = {elem->getFESupport()[j].second[0], elem->getFESupport()[j].second[1]};
	nodeIDxs[j] = elem->getFESupport()[j].first;
      }
      LagrangianBasis<2,1> basis(nodes);

      // evaluate the solution
      for(size_t j = 0; j < 3; ++j){
	solValue += solLaplace[nodeIDxs[j]] * basis.getBasisElement(j)(point);
      }
      solVect.push_back(solValue);
    }else{
      solVect.push_back(std::numeric_limits<double>::quiet_NaN());
    }

    if(jj < xC[0].size()){
      ev_row.push_back(solValue);
      ++jj;
    }else{
      ev.push_back(ev_row);
      ev_row.erase(ev_row.begin(), ev_row.end());
      jj = 0;
      ev_row.push_back(solValue);
      ++jj;
    }
  }
  ev.push_back(ev_row);

  matplotlibcpp::plot_surface(xC,yC,ev);
  matplotlibcpp::show();  */
  // ### END OF EVALUATION

  //   auto evalSolutionLaplace = [&tree, &solLaplace](const SVector<2>& point) mutable -> double {
  //   // search element containing point
  //   std::shared_ptr<Element<2,2>> elem = tree.search(point);

  //   double solValue = 0;

  //   if(elem == nullptr) std::cout << "elemento non trovato" << std::endl;
  //   else{
  //     std::cout << elem->getID() << std::endl;
  //     // build a lagrangian basis over the element
  //     std::array<std::array<double, 2>, 3> nodes{};
  //     std::array<unsigned int, 3> nodeIDxs{};
  //     for(size_t j = 0; j < 3; ++j){ // 3 deve scalare sul numero di nodi
  // 	nodes[j] = {elem->getFESupport()[j].second[0], elem->getFESupport()[j].second[1]};
  // 	//std::cout << elem->getCoords()[j] << std::endl;
  // 	//std::cout << "----" << std::endl;
  // 	nodeIDxs[j] = elem->getFESupport()[j].first;
  //     }
  //     LagrangianBasis<2,1> basis(nodes);
    
  //     // evaluate the solution
  //     for(size_t j = 0; j < 3; ++j){
  // 	solValue += solLaplace[nodeIDxs[j]] * basis.getBasisElement(j)(elem->getBaryMatrix()*point);
  // 	std::cout << "node: " << nodeIDxs[j] << std::endl;
  // 	std::cout << "coefficient:  " << solLaplace[nodeIDxs[j]] << std::endl;
  // 	std::cout << "eval of basis: " << basis.getBasisElement(j)(point) << std::endl;

  //       std::cout << solLaplace[nodeIDxs[j]] * basis.getBasisElement(j)(point) << std::endl;
  //     }
  //   }
    
  //   return solValue;
  // };
  
  //   std::cout << "eval laplace: " << evalSolutionLaplace(SVector<2>(0.4, 0.4)) << std::endl;  

  
  /*  SVector<2> p(1,1);

  // check if integral of gradient is correct
  std::function<double(SVector<2>)> f1 = [](SVector<2> x) -> double { return 1; };
  std::function<double(SVector<2>)> f2 = [](SVector<2> x) -> double { return 1; };

  std::array<std::function<double(SVector<2>)>, 2> arr = {f1, f2};
  VectorField<2> vfield(arr);

  auto pp = (vfield + vfield).dot(p);

  auto vv = vfield + vfield;
  auto xx = vv[0];
  
  std::cout << pp(SVector<2>(0,0)) << std::endl;

  auto g = Gradient(SVector<2>(1,1)) + Laplacian();*/


  
  //  std::cout << "num nodes: " << multiTree.treeView(1).getNode(1)->getChildren()[fdaPDE::core::LEFT]->getData() << std::endl;
  
  //std::function<void(std::string)> functorNode = [](std::string x) -> void { std::cout <<  x << std::endl; };
  /*
  Tree<std::string> t("02");
  t.insert("01");
  t.insert("11");
  t.insert("00");
  t.insert("10");
  t.insert("10");
  t.insert("20");

  //t.DFS(functorNode);

  std::vector<double> knots = {0, 1, 2, 3, 4, 5};
  SplineBasis b(knots, 3);

  std::function<void(SplineNode)> functorInt = [](SplineNode x) -> void { std::cout << x.getKnotSpan().first << " - " << x.getKnotSpan().second << std::endl; };
  //b[0].getTree().DFS(functorInt);

  std::vector<double> xx{};
  std::vector<double> xx0{};
  std::vector<double> xx1{};
  std::vector<double> xx2{};
  std::vector<double> xx3{};
  std::vector<double> xx4{};
  std::vector<double> xx5{};
  std::vector<double> xx6{};
  std::vector<double> xx7{};  
  std::vector<double> dd{};  
  
  
  //std::cout << b[0](1.4) << std::endl;

 
  for(double x = 0; x < 5; x += 0.05){
    xx.push_back(x);
    xx0.push_back(b[0](x));
    xx1.push_back(b[1](x));
    xx2.push_back(b[2](x));
    xx3.push_back(b[3](x));
    xx4.push_back(b[4](x));
    xx5.push_back(b[5](x));
    xx6.push_back(b[6](x));
    xx7.push_back(b[7](x));
    
    //    dd.push_back (gradSpline(x));
    //std::cout << gradSpline(x) << std::endl;
    //xx2.push_back(b[2](x));
  }

  matplotlibcpp::plot(xx, xx0);
  matplotlibcpp::plot(xx, xx1);
  matplotlibcpp::plot(xx, xx2);
  matplotlibcpp::plot(xx, xx3);
  matplotlibcpp::plot(xx, xx4);
  matplotlibcpp::plot(xx, xx5);
  matplotlibcpp::plot(xx, xx6);
  matplotlibcpp::plot(xx, xx7);
  //matplotlibcpp::plot(xx, dd);
  matplotlibcpp::show();
 
  // print spline derivative...
  // auto gradSpline0 = b[0].gradient().gradient();
  // auto gradSpline1 = b[1].gradient().gradient();
  // auto gradSpline2 = b[2].gradient().gradient();
  // auto gradSpline3 = b[3].gradient().gradient();
  // auto gradSpline4 = b[4].gradient().gradient();
  // auto gradSpline5 = b[5].gradient().gradient();
  // auto gradSpline6 = b[6].gradient().gradient();
  // auto gradSpline7 = b[7].gradient().gradient();

  // for(double x = 0; x < 5; x += 0.05){
  //   xx.push_back(x);
  //   xx0.push_back(gradSpline0(x));
  //   xx1.push_back(gradSpline1(x));
  //   xx2.push_back(gradSpline2(x));
  //   xx3.push_back(gradSpline3(x));
  //   xx4.push_back(gradSpline4(x));
  //   xx5.push_back(gradSpline5(x));
  //   xx6.push_back(gradSpline6(x));
  //   xx7.push_back(gradSpline7(x));
    
  //   //    dd.push_back (gradSpline(x));
  //   //std::cout << gradSpline(x) << std::endl;
  //   //xx2.push_back(b[2](x));
    
  // }

  // matplotlibcpp::plot(xx, xx0);
  // matplotlibcpp::plot(xx, xx1);
  // matplotlibcpp::plot(xx, xx2);
  // matplotlibcpp::plot(xx, xx3);
  // matplotlibcpp::plot(xx, xx4);
  // matplotlibcpp::plot(xx, xx5);
  // matplotlibcpp::plot(xx, xx6);
  // matplotlibcpp::plot(xx, xx7);
  // // //matplotlibcpp::plot(xx, dd);
  // matplotlibcpp::show();
  
  */

  /*  std::tuple<Gradient<1>> a;
  std::tuple<Identity<>> b;

  auto c = std::tuple_cat(a,b);

  auto formA = Laplacian() + dot(SVector<2>(2,1), Gradient()) - 100*Identity();

  auto type = formA.getTypeList();
  
  if constexpr(is_symmetric<decltype(formA)>::value){
    std::cout << "forma bilineare simmetrica" << std::endl;
  }else{
    std::cout << "forma bilineare non simmetrica" << std::endl;
  }

  if constexpr(is_elliptic<decltype(formA)>::value){
    std::cout << "operatore ellittico" << std::endl;
  }else{
    std::cout << "operatore non ellittico" << std::endl;
  }

  */

  // parabolic solver

  Mesh<2,2> m("circle_points.csv", "circle_edges.csv", "circle_triangles.csv", "circle_neighbors.csv", "circle_boundary_markers.csv");
  std::cout << "mesh info loaded..." << std::endl;

  CSVReader reader;
  CSVFile<double> boundary_data = reader.parseFile<double>("circle_data.csv");
  Eigen::Matrix<double, Eigen::Dynamic, 1> boundaryMatrix = boundary_data.toEigen();

  SMatrix<2> K { {10,0}, {0,5} };
  
  auto form = dT() + Laplacian() + dot(SVector<2>(10,5), Gradient()); // - 100*Identity();
  
  std::function<double(SVector<2>)> forcing = [](SVector<2> x) -> double { return 4; };
  ScalarField<2> forcing_field(forcing);

  DMatrix forcingData;
  forcingData.resize(m.getNumberOfElements(), 20);
  forcingData.fill(0);

  PDE problem(m, form, forcingData);
  problem.setDirichletBC(forcingData);

  DVector initialData;
  initialData.resize(m.getNumberOfNodes());
  initialData.fill(0);

  initialData[150] = 1;
  //  initialData[150] = 1;
  
  problem.setInitialCondition(initialData);
  
  //  std::cout << problem.getForcingData();
  
  LagrangianBasis<2, 1> basis{};  // define functional basis
  Integrator<2, 6> integrator{};  // define integrator

  // define solver
  //FEMStandardSpaceTimeSolver solver(basis, integrator);
  //solver.solve(problem, 10.0, 0.1);
  
  problem.solve(basis, integrator, 0.01);
  
  for(std::size_t ii = 0; ii < 19; ++ii){
    DVector sol = problem.getSolution().col(ii);

    std::cout << "assemblato, sarà corretto?? :|" << std::endl;
  
    // std::cout << sol << std::endl;
  
    // std::cout << sol << std::endl;
    fdaPDE::core::MESH::ADT<2,2> tree(m);
  
    auto evalSolution = [&tree, &sol](const SVector<2>& point) mutable -> double {
      // search element containing point
      std::shared_ptr<Element<2,2>> elem = tree.search(point);

      double solValue = 0;
    
      if(elem == nullptr) std::cout << "elemento non trovato" << std::endl;
      else{
	std::cout << "elementoID: " << elem->getID() << std::endl;
	// build a lagrangian basis over the element
	std::array<unsigned int, 3> nodeIDxs{};
	for(size_t j = 0; j < 3; ++j){ // 3 deve scalare sul numero di nodi
	  nodeIDxs[j] = elem->getFESupport()[j].first;
	}
	LagrangianBasis<2, 1> basis(*elem);

	// evaluate the solution
	for(size_t j = 0; j < 3; ++j){
	  solValue += (sol[nodeIDxs[j]]) * basis[j](point);
	  std::cout << "node: " << nodeIDxs[j] << std::endl;
	  std::cout << "coefficient:  " << sol[nodeIDxs[j]] << std::endl;
	  std::cout << "eval of basis: " << basis[j](point) << std::endl;
	  std::cout << sol[nodeIDxs[j]] * basis[j](point) << std::endl;
	}
      }
      std::cout << "mo te la printo" << std::endl;
      return solValue;
    };
  
    std::cout << "valutazione soluzione: \n\n" << evalSolution(SVector<2>(0.01,0.01)) << std::endl;

    //  std::cout << sol << std::endl;
  
    // #### SOLUTION EVALUATION
    std::vector<SVector<2>> listPoint;  

    std::vector<double> solVect{};

    std::vector<std::vector<double>> ev, xC, yC; // plot vectors

    for(double x = m.getMeshRange()[0].first - 0.2; x <= m.getMeshRange()[0].second + 0.2; x+=0.05){
      std::vector<double> x_row, y_row;
      for(double y = m.getMeshRange()[1].first - 0.2; y <= m.getMeshRange()[1].second + 0.2; y+=0.05){
	listPoint.push_back(SVector<2>(x,y));

	x_row.push_back(x); // plot
	y_row.push_back(y);

      }
    
      xC.push_back(x_row); // plot
      yC.push_back(y_row);
    }
  
    int jj = 0; // plot
    std::vector<double> ev_row;
    for(const SVector<2>& point : listPoint){
      // search element containing point
      std::shared_ptr<Element<2,2>> elem = tree.search(point);

      double solValue = 0;
      if(elem != nullptr){
	// build a lagrangian basis over the element
	std::array<std::array<double, 2>, 3> nodes{};
	std::array<unsigned int, 3> nodeIDxs{};
	for(size_t j = 0; j < 3; ++j){ // 3 deve scalare sul numero di nodi
	  nodes[j] = {elem->getFESupport()[j].second[0], elem->getFESupport()[j].second[1]};
	  nodeIDxs[j] = elem->getFESupport()[j].first;
	}
	LagrangianBasis<2,1> basis(nodes);

	// evaluate the solution
	for(size_t j = 0; j < 3; ++j){
	  solValue += (sol[nodeIDxs[j]]) * basis[j](point);
	}
	solVect.push_back(solValue);
      }else{
	solVect.push_back(std::numeric_limits<double>::quiet_NaN());
      }

      if(jj < xC[0].size()){
	ev_row.push_back(solValue);
	++jj;
      }else{
	ev.push_back(ev_row);
	ev_row.erase(ev_row.begin(), ev_row.end());
	jj = 0;
	ev_row.push_back(solValue);
	++jj;
      }
    }
    ev.push_back(ev_row);

    matplotlibcpp::plot_surface(xC,yC,ev);
    matplotlibcpp::title("dT() + Laplacian() + Gradient([10, 5]) = 4");
    matplotlibcpp::show();
    // ### END OF EVALUATION
  }
  
  return 0;
}
