// QR decomposition (using classical GS)
template <typename MatrixType>
class QRDecomposition {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, "QR is only defined for square matrices");

public:
    static constexpr int N = MatrixType::Rows;
    using Scalar  = typename MatrixType::Scalar;
    using MatrixN = Matrix<Scalar, N, N>;

    constexpr QRDecomposition() = default;

    template <typename Xpr>
    constexpr explicit QRDecomposition(const SquareMatrixBase<N, Xpr>& m) {
        static_assert(std::is_same_v<Scalar, typename Xpr::Scalar>, "QR: scalar types must match");
        compute(m);
    }

    template <typename Xpr>
    constexpr void compute(const SquareMatrixBase<N, Xpr>& xpr) {

        MatrixView<Scalar, N, N, Xpr::StorageOrder, true> A(xpr.derived());
        R_.setZero();

        // MGS with basis completion
        Q_ = modified_gram_schmidt<decltype(A)>(A);

        // coefficients matrix
        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                R_(i,j) = Q_.col(i).dot(A.col(j));
            }
        }
    }

    constexpr auto Q() const { return Q_; }
    constexpr auto R() const { return R_; }
    [[nodiscard]] bool info() const { return info_; }

private:
    OrthogonalMatrix<Scalar, N> Q_;
    UpperTriangularMatrix<Scalar, N> R_;
    bool info_ = Success;
};
