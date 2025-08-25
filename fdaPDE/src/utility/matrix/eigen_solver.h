// eigen decomposition
template <typename MatrixType, int Solver_ = Best>
class EigenDecomposition {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, "EVD is only defined for square matrices");
    fdapde_static_assert(MatrixType::Rows >= 2, "EVD: size must be >= 2 for analytic specializations");

public:
    static constexpr int N = MatrixType::Rows;
    using Scalar  = typename MatrixType::Scalar;
    static constexpr int StorageOrder = MatrixType::StorageOrder;
    static constexpr int NestAsRefBit = MatrixType::NestAsRefBit;
    using VectorType = Vector<Scalar, N, NestAsRefBit>;
    using MatrixN = Matrix<Scalar, N, N, StorageOrder, NestAsRefBit>;
    static constexpr int Solver = Solver_;

    constexpr EigenDecomposition() = default;

    template <typename Xpr>
    constexpr explicit EigenDecomposition(const SquareMatrixBase<N, Xpr>& m) {
        static_assert(std::is_same_v<Scalar, typename Xpr::Scalar>, "EVD: scalar types must match");
        fdapde_static_assert(internals::is_symmetric_v<Xpr>, "EVD compute requires symmetric input expression");
        compute(m);
    }

    template <typename Xpr>
    constexpr void compute(const SquareMatrixBase<N, Xpr>& xpr) {
        static_assert(std::is_same_v<Scalar, typename Xpr::Scalar>, "EVD: scalar types must match");
        fdapde_static_assert(internals::is_symmetric_v<Xpr>, "EVD compute requires symmetric input expression");

        // Generic fallback: simple QR iteration
        MatrixN A(xpr.derived());
        QRDecomposition<decltype(A)> qr;
        MatrixN V(MatrixN::Identity());
        Scalar tol = N * std::numeric_limits<Scalar>::epsilon() * A.norm();
        int iter = 0;
        for (; iter < 100; ++iter) {
            qr.compute(A);
            A = MatrixN(qr.R() * qr.Q());
            V =  MatrixN(V * qr.Q());
            if (A.off_diagonal_norm() < tol) break;
        }
        for (int i = 0; i < N; ++i) eigenvalues_[i] = A(i,i);
        eigenvectors_ = V;
    }

    constexpr auto& eigenvalues() const  { return eigenvalues_; }
    constexpr auto& eigenvectors() const { return eigenvectors_; }

private:
    OrthogonalMatrix<Scalar, N> eigenvectors_{};
    VectorType eigenvalues_{};
};
