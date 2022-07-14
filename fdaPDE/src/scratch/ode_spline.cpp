Triangulation<1, 1> T = Triangulation<1, 1>::Interval(0, 1, 21);
  BsSpace Vh(T, 3);
 TrialFunction f(Vh);
  TestFunction v(Vh); 
  auto a = integral(T)(dxx(f) * dxx(v));
  ScalarField<1, decltype([](const Eigen::Matrix<double, 1, 1>& p) { return 0; })> u;
  auto F = integral(T)(u * v);

  Eigen::SparseMatrix<double> A = a.assemble();
  Eigen::Matrix<double, Dynamic, 1> b = F.assemble();

  Eigen::SparseLU<Eigen::SparseMatrix<double>> invA(A);
  invA.solve(b);