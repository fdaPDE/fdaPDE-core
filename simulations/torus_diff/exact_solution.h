#include "fields.h"


namespace diff_torus {

constexpr int M = 3;
using Vec = Eigen::Matrix<double, M, 1>;
using Fun = std::function<double(const Vec&)>;

constexpr int m = 2, n = 2;
constexpr double R = 2.0;
constexpr double r = 1.0;

// Exact solution
inline fdapde::ScalarField<M> make_u_exact() {
    return fdapde::ScalarField<M>([](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double phi = std::atan2(y,x);
            double theta = std::atan2(z, std::sqrt(x*x + y*y) - R);
            return std::sin(m * theta) * std::sin(n * phi);
    });
}

// RHS of the Poisson equation
inline fdapde::ScalarField<M> make_rhs() {
    return fdapde::ScalarField<M>([](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double phi = std::atan2(y,x);
            double theta = std::atan2(z, std::sqrt(x*x + y*y) - R);

            double denom = R + r * std::cos(theta);

            return (m * std::sin(n * phi) * std::cos(m * theta) * std::sin(theta) / (r * denom)) +
                   std::sin(n * phi) * std::sin(m * theta) * (m*m / (r*r) + n*n / (denom * denom));
    });
}

// Gradient of the exact solution
inline fdapde::VectorField<M, M, Fun> make_grad_u_exact() {
    fdapde::VectorField<M, M, Fun> df;

        df(0, 0) = [](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);

            double phi = std::atan2(y, x);
            double rho = std::sqrt(x * x + y * y);
            double theta = std::atan2(z, rho - R);

            double sin_phi = std::sin(phi), cos_phi = std::cos(phi);
            double sin_theta = std::sin(theta), cos_theta = std::cos(theta);

            double dphi = n * std::cos(n * phi) * std::sin(m * theta);
            double dtheta = m * std::cos(m * theta) * std::sin(n * phi);

            double denom_phi = (R + r * cos_theta) * (R + r * cos_theta);
            double denom_theta = r * r;

            double denom = R + r * cos_theta;

            // ∂Φ/∂phi (X component)
            double dphi_x = -denom * sin_phi;

            // ∂Φ/∂theta (X component)
            double dtheta_x = -r * sin_theta * cos_phi;

            return (dphi / denom_phi) * dphi_x + (dtheta / denom_theta) * dtheta_x;
        };

        df(1, 0) = [](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);

            double phi = std::atan2(y, x);
            double rho = std::sqrt(x * x + y * y);
            double theta = std::atan2(z, rho - R);

            double sin_phi = std::sin(phi), cos_phi = std::cos(phi);
            double sin_theta = std::sin(theta), cos_theta = std::cos(theta);

            double dphi = n * std::cos(n * phi) * std::sin(m * theta);
            double dtheta = m * std::cos(m * theta) * std::sin(n * phi);

            double denom_phi = (R + r * cos_theta) * (R + r * cos_theta);
            double denom_theta = r * r;

            double denom = R + r * cos_theta;

            // ∂Φ/∂phi (Y component)
            double dphi_y = denom * cos_phi;

            // ∂Φ/∂theta (Y component)
            double dtheta_y = -r * sin_theta * sin_phi;

            return (dphi / denom_phi) * dphi_y + (dtheta / denom_theta) * dtheta_y;
        };

        df(2, 0) = [](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);

            double phi = std::atan2(y, x);
            double rho = std::sqrt(x * x + y * y);
            double theta = std::atan2(z, rho - R);

            double sin_phi = std::sin(phi);
            double cos_theta = std::cos(theta);

            double dtheta = m * std::cos(m * theta) * std::sin(n * phi);
            double denom_theta = r * r;

            // ∂Φ/∂theta (Z component)
            double dtheta_z = r * cos_theta;

            return (dtheta / denom_theta) * dtheta_z;
        };
    return df;
}

} // namespace ring