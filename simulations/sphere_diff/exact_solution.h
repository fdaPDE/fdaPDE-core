#include "fields.h"


namespace diff_sphere {

constexpr int M = 3;
using Vec = Eigen::Matrix<double, M, 1>;
using Fun = std::function<double(const Vec&)>;

constexpr int alpha = 4;
constexpr int beta = 5;
constexpr int lambda = 3;

// Exact solution
inline fdapde::ScalarField<M> make_u_exact() {
    return fdapde::ScalarField<M>([](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double r = std::sqrt(x * x + y * y + z * z);
            double phi = std::atan2(y, x);
            double theta = std::acos(z / r);
            return std::sin(beta * theta) * std::sin(alpha * phi);

    });
}

// RHS of the Poisson equation
inline fdapde::ScalarField<M> make_rhs() {
    return fdapde::ScalarField<M>([](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double r = std::sqrt(x * x + y * y + z * z);
            double phi = std::atan2(y, x);
            double theta = std::acos(z / r);
            return std::sin(alpha * phi) * std::sin(beta * theta) *( alpha*alpha/(std::sin(theta)*std::sin(theta)) +
                 beta*beta  - beta*((std::cos(theta) * std::cos(beta * theta))/(std::sin(theta) * std::sin(beta * theta))));
    });
}

// Gradient of the exact solution
inline fdapde::VectorField<M, M, Fun> make_grad_u_exact() {
    fdapde::VectorField<M, M, Fun> df;

        df(0, 0) = [](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double r = std::sqrt(x * x + y * y + z * z);
            double phi = std::atan2(y, x);
            double theta = std::acos(z / r);
        
            double df_dtheta = beta * std::cos(beta * theta) * std::sin(alpha * phi);
            double df_dphi = alpha * std::cos(alpha * phi) * std::sin(beta * theta);
            double sintheta = std::sin(theta);
        
            return df_dtheta * std::cos(theta) * std::cos(phi)
                 - (df_dphi / sintheta) * std::sin(phi);
        };

        df(1, 0) = [](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double r = std::sqrt(x * x + y * y + z * z);
            double phi = std::atan2(y, x);
            double theta = std::acos(z / r);
        
            double df_dtheta = beta * std::cos(beta * theta) * std::sin(alpha * phi);
            double df_dphi = alpha * std::cos(alpha * phi) * std::sin(beta * theta);
            double sintheta = std::sin(theta);
        
            return df_dtheta * std::cos(theta) * std::sin(phi)
                 + (df_dphi / sintheta) * std::cos(phi);
        };

        df(2, 0) = [](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double r = std::sqrt(x * x + y * y + z * z);
            double phi = std::atan2(y, x);
            double theta = std::acos(z / r);
        
            double df_dtheta = beta * std::cos(beta * theta) * std::sin(alpha * phi);
            return -df_dtheta * std::sin(theta);
        };
    return df;
}

} // namespace ring