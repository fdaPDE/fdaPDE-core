#include "fields.h"


namespace advdiff_ring {

constexpr int M = 2;
using Vec = Eigen::Matrix<double, M, 1>;
using Fun = std::function<double(const Vec&)>;

// Exact solution
inline fdapde::ScalarField<M> make_u_exact() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        return x * x * y + y * y * y;
    });
}

// RHS of the Poisson equation
inline fdapde::ScalarField<M> make_rhs() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        return - 20.0 * y - 4.0 * x;
    });
}

// Gradient of the exact solutionadd things here
inline fdapde::VectorField<M, M, Fun> make_grad_u_exact() {
    fdapde::VectorField<M, M, Fun> df;

    df(0, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        return 2.0 * x * y;
    };

    df(1, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        return x * x + 3.0 * y * y;
    };

    return df;
}


inline fdapde::ScalarField<M> make_g_neumann() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        constexpr double eps = 1e-8;
        double dudx = 2.0 * x * y;
        double dudy = x * x + 3.0 * y * y;
        double flux = 0.0;

        if (std::abs(x - 1.0) < eps) {
            flux = 4.0 * dudx + dudy;
        } else if (std::abs(x - 0.0) < eps) {
            flux = -(4.0 * dudx + dudy);
        } else if (std::abs(y - 0.0) < eps) {
            flux = -1.0 * (dudx + 2.0 * dudy);
        } else if (std::abs(y - 1.0) < eps) {
            flux = dudx + 2.0 * dudy;
        }

        return flux;
    });
}

/*

inline fdapde::ScalarField<M> make_g_neumann() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        if(x == 0 || y == 0) {
            return 0.0; // Neumann condition at the origin
        } else {
            return 2.0; // Some constant value for other points
        }
    });
}
    */

} // namespace ring