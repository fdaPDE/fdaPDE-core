#include "fields.h"


namespace advdiff_ring {

constexpr int M = 2;
using Vec = Eigen::Matrix<double, M, 1>;
using Fun = std::function<double(const Vec&)>;

// Exact solution
inline fdapde::ScalarField<M> make_u_exact() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        return std::sin(M_PI * x) * std::cos(2 * M_PI * y);
    });
}

// RHS of the Poisson equation
inline fdapde::ScalarField<M> make_rhs() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        return 14 * M_PI * M_PI * std::sin(M_PI * x) * std::cos(2 * M_PI * y)
               + 4 * M_PI * M_PI * std::cos(M_PI * x) * std::sin(2 * M_PI * y);
    });
}

// Gradient of the exact solutionadd things here
inline fdapde::VectorField<M, M, Fun> make_grad_u_exact() {
    fdapde::VectorField<M, M, Fun> df;

    df(0, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        return M_PI * std::cos(M_PI * x) * std::cos(2 * M_PI * y);
    };

    df(1, 0) = [](const Vec& p) {
        double x = p(0), y = p(1);
        return -2 * M_PI * std::sin(M_PI * x) * std::sin(2 * M_PI * y);
    };

    return df;
}


inline fdapde::ScalarField<M> make_g_neumann() {
    return fdapde::ScalarField<M>([](const Vec& p) {
        double x = p(0), y = p(1);
        constexpr double eps = 1e-8;
        double dudx = M_PI * std::cos(M_PI * x) * std::cos(2 * M_PI * y);
        double dudy = -2 * M_PI * std::sin(M_PI * x) * std::sin(2 * M_PI * y);
        double flux = 0.0;

        if (std::abs(x - 0.0) < eps) {
            flux = -(2 * dudx + dudy); // n = (-1,0)
        } else if (std::abs(x - 1.0) < eps) {
            flux =  (2 * dudx + dudy); // n = (1,0)
        } else if (std::abs(y - 0.0) < eps) {
            flux = -(dudx + 3 * dudy); // n = (0,-1)
        } else if (std::abs(y - 1.0) < eps) {
            flux =  (dudx + 3 * dudy); // n = (0,1)
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