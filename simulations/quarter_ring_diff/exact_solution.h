#include "fields.h"


namespace diff_ring {

constexpr int M = 2;
using Vec = Eigen::Matrix<double, M, 1>;
using Fun = std::function<double(const Vec&)>;

// Exact solution
inline fdapde::ScalarField<M> make_u_exact() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        return 2 * x * y * (x * x + y * y - 1) * (4 - x * x - y * y);
    });
}

// RHS of the Poisson equation
inline fdapde::ScalarField<M> make_rhs() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        return 8 * x * y * (8 * x * x + 8 * y * y - 15);
    });
}

// Gradient of the exact solution
inline fdapde::VectorField<M, M, Fun> make_grad_u_exact() {
    fdapde::VectorField<M, M, Fun> df;

    df(0, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        return -2 * y * (5 * std::pow(x, 4) + 6 * x * x * y * y - 15 * x * x + std::pow(y, 4) - 5 * y * y + 4);
    };

    df(1, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        return -2 * x * (std::pow(x, 4) + 6 * x * x * y * y - 5 * x * x + 5 * std::pow(y, 4) - 15 * y * y + 4);
    };

    return df;
}

} // namespace ring