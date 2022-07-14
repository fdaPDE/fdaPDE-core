#include "fields.h"


namespace bih_sphere {

constexpr int M = 3;
using Vec = Eigen::Matrix<double, M, 1>;
using Fun = std::function<double(const Vec&)>;

constexpr int lambda = 3;

// Exact solution
inline fdapde::ScalarField<M> make_u_exact() {
    return fdapde::ScalarField<M>([](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double r = std::sqrt(x * x + y * y + z * z);
            double phi = std::atan2(y, x);
            double theta = std::acos(z / r);
            return std::sin(lambda * phi) * std::pow(std::sin(theta), lambda);

    });
}

// RHS of the biharmonic equation
inline fdapde::ScalarField<M> make_rhs() {
    return fdapde::ScalarField<M>([](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double r = std::sqrt(x * x + y * y + z * z);
            double phi = std::atan2(y, x);
            double theta = std::acos(z / r);
            return lambda * lambda *(lambda + 1) * (lambda + 1) * std::sin(lambda * phi) * std::pow(std::sin(theta),lambda) ;
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
        
            double df_dtheta = lambda *std::cos(theta) * std::sin(lambda * phi) * std::pow(std::sin(theta), lambda - 1);
            double df_dphi = lambda * std::cos(lambda * phi) * std::pow(std::sin(theta), lambda - 1);
        
            return (x * z * df_dtheta - y * df_dphi)/std::sqrt(x*x + y*y);
        };

        df(1, 0) = [](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double r = std::sqrt(x * x + y * y + z * z);
            double phi = std::atan2(y, x);
            double theta = std::acos(z / r);
        
            double df_dtheta = lambda *std::cos(theta) * std::sin(lambda * phi) * std::pow(std::sin(theta), lambda - 1);
            double df_dphi = lambda * std::cos(lambda * phi) * std::pow(std::sin(theta), lambda - 1);
        
            return (y * z * df_dtheta + x * df_dphi)/std::sqrt(x*x + y*y);
        };

        df(2, 0) = [](const Vec& p) {
            double x = p(0), y = p(1), z = p(2);
            double r = std::sqrt(x * x + y * y + z * z);
            double phi = std::atan2(y, x);
            double theta = std::acos(z / r);
        
            double df_dtheta = lambda *std::cos(theta) * std::sin(lambda * phi) * std::pow(std::sin(theta), lambda - 1);
            double df_dphi = lambda * std::cos(lambda * phi) * std::pow(std::sin(theta), lambda - 1);
            return -df_dtheta * std::sqrt(x*x + y*y) ;
        };
    return df;
}

// Manifold Hessian (Hessian–Beltrami) of the exact solution expressed in Cartesian coordinates
inline fdapde::MatrixField<M, M, M> make_hessian_u_exact() {
    fdapde::MatrixField<M, M, M> hess;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            hess(i, j) = [i, j](const Vec& p) {
                // u(x, y, z) = y
                double x = p(0);
                double y = p(1);
                double phi = std::atan2(y, x);
                // Hessian in ambient coordinates
                Eigen::Matrix<double, M, M> D = Eigen::Matrix<double, M, M>::Zero();
                D(0,0) = 6*y;
                D(0,1) = 6*x;
                D(1,0) = 6*x;
                D(1,1) = - 6*y;

                // Projection operator
                Eigen::Matrix<double, M, M> I = Eigen::Matrix<double, M, M>::Identity();
                Eigen::Matrix<double, M, M> P = I - p * p.transpose();

                // Gradient of u = 3y(x^2 + y^2) - 4y^3
                Eigen::Matrix<double, M, 1> grad;
                grad(0) = 6 * x * y;
                grad(1) = 3*x*x-3*y*y;
                grad(2) = 0.0;

                double gdotx = grad.dot(p);

                // Manifold Hessian with second fundamental form correction
                Eigen::Matrix<double, M, M> H = P * D * P - gdotx * P;

                return H(i, j);
            };
        }
    }

    return hess;
}

} // namespace ring