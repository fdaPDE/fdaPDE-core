#include "fields.h"


namespace bih_ring {

constexpr int M = 2;
using Vec = Eigen::Matrix<double, M, 1>;
using Fun = std::function<double(const Vec&)>;

// Exact solution
inline fdapde::ScalarField<M> make_u_exact() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        double r2 = x*x + y*y;
        return x*x * y*y * std::pow(r2 - 1, 2) * std::pow(r2 - 4, 2);
    });
}

// RHS of the Poisson equation
inline fdapde::ScalarField<M> make_rhs() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        return 8 * (
                    16 + 57 * std::pow(x, 8) - 360 * std::pow(y, 2) + 693 * std::pow(y, 4)
                    - 370 * std::pow(y, 6) + 57 * std::pow(y, 8)
                    + 2 * std::pow(x, 6) * (-185 + 786 * std::pow(y, 2))
                    + std::pow(x, 4) * (693 - 6150 * std::pow(y, 2) + 3030 * std::pow(y, 4))
                    + 6 * std::pow(x, 2) * (-60 + 891 * std::pow(y, 2) - 1025 * std::pow(y, 4) + 262 * std::pow(y, 6))
                );
    });
}

// Gradient of the exact solution
inline fdapde::VectorField<M, M, Fun> make_grad_u_exact() {
    fdapde::VectorField<M, M, Fun> df;

    df(0, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        return 2 * x * y * y * (x*x + y*y - 4) * (x*x + y*y - 1) *
               (4 + 5*std::pow(x, 4) - 5*y*y + std::pow(y, 4) + 3*x*x*(-5 + 2*y*y));
    };

    df(1, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        return 2 * x * x * y * (x*x + y*y - 4) * (x*x + y*y - 1) *
               (4 + std::pow(x, 4) - 15*y*y + 5*std::pow(y, 4) + x*x*(-5 + 6*y*y));
    };

    return df;
}

inline fdapde::MatrixField<M, M, M> make_hessian_u_exact() {
    fdapde::MatrixField<M, M, M> hess;

    hess(0, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        double r2 = x * x + y * y;
        return 2 * y * y * (
            16 * std::pow(x, 4) * (r2 - 4) * (r2 - 1) +
            8 * x * x * std::pow(r2 - 4, 2) * (r2 - 1) +
            2 * x * x * std::pow(r2 - 4, 2) * (3 * x * x + y * y - 1) +
            8 * x * x * (r2 - 4) * std::pow(r2 - 1, 2) +
            2 * x * x * std::pow(r2 - 1, 2) * (3 * x * x + y * y - 4) +
            std::pow(r2 - 4, 2) * std::pow(r2 - 1, 2)
        );
    };

    hess(0, 1) = hess(1, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        double r2 = x * x + y * y;
        return 4 * x * y * (
            2 * x * x * y * y * std::pow(r2 - 4, 2) +
            8 * x * x * y * y * (r2 - 4) * (r2 - 1) +
            2 * x * x * y * y * std::pow(r2 - 1, 2) +
            2 * x * x * std::pow(r2 - 4, 2) * (r2 - 1) +
            2 * x * x * (r2 - 4) * std::pow(r2 - 1, 2) +
            2 * y * y * std::pow(r2 - 4, 2) * (r2 - 1) +
            2 * y * y * (r2 - 4) * std::pow(r2 - 1, 2) +
            std::pow(r2 - 4, 2) * std::pow(r2 - 1, 2)
        );
    };

    hess(1, 1) = [](const Vec& p) {
        double x = p(0), y = p(1);
        double r2 = x * x + y * y;
        return 2 * x * x * (
            16 * std::pow(y, 4) * (r2 - 4) * (r2 - 1) +
            8 * y * y * std::pow(r2 - 4, 2) * (r2 - 1) +
            2 * y * y * std::pow(r2 - 4, 2) * (x * x + 3 * y * y - 1) +
            8 * y * y * (r2 - 4) * std::pow(r2 - 1, 2) +
            2 * y * y * std::pow(r2 - 1, 2) * (x * x + 3 * y * y - 4) +
            std::pow(r2 - 4, 2) * std::pow(r2 - 1, 2)
        );
    };

    return hess;
}

} // namespace ring