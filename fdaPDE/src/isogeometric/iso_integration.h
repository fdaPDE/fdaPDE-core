#ifndef __ISO_INTEGRATION_H__
#define __ISO_INTEGRATION_H__

#include "header_check.h"


namespace fdapde {
namespace internals{

struct iso_quadrature_gauss_base { };
template <typename T>
concept is_iso_quadrature_gauss = std::is_base_of_v<iso_quadrature_gauss_base, T>;
template <typename T> [[maybe_unused]] static constexpr bool is_iso_quadrature_gauss_v = is_iso_quadrature_gauss<T>;

// Higher degree quadrature
template <typename LhsQuadrature, typename RhsQuadrature>
requires(
    LhsQuadrature::local_dim == RhsQuadrature::local_dim && is_iso_quadrature_gauss_v<LhsQuadrature> &&
    is_iso_quadrature_gauss_v<RhsQuadrature>)
struct higher_degree_iso_quadrature :
    std::type_identity<
    std::conditional_t<(LhsQuadrature::order > RhsQuadrature::order), LhsQuadrature, RhsQuadrature>> { };

template <typename LhsQuadrature, typename RhsQuadrature>
using higher_degree_iso_quadrature_t = higher_degree_iso_quadrature<LhsQuadrature, RhsQuadrature>::type;

// quadrature points and weights at: https://people.sc.fsu.edu/~jburkardt/datasets/datasets.html
template <int LocalDim, int Size> struct iso_quadrature_gauss_legendre;

// 1D 1 point formula
template <> struct iso_quadrature_gauss_legendre<1, 1> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 1;
    static constexpr int order  = 1;
    static constexpr int degree = 2 * order - 1;   // 1

    static constexpr Vector<double, order> nodes {
        std::array<double, order> {0.000000000000000}
    };
    static constexpr Vector<double, order> weights {
        std::array<double, order> {2.000000000000000}
    };
};

// 1D 2 point formula
template <> struct iso_quadrature_gauss_legendre<1, 2> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 1;
    static constexpr int order  = 2;
    static constexpr int degree = 2 * order - 1;   // 3

    static constexpr Vector<double, order> nodes {
        std::array<double, order> {-0.5773502691896257, 0.5773502691896257}
    };
    static constexpr Vector<double, order> weights {
        std::array<double, order> { 0.9999999999999998, 0.9999999999999998}
    };
};

// 1D 3 point formula
template <> struct iso_quadrature_gauss_legendre<1, 3> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 1;
    static constexpr int order  = 3;
    static constexpr int degree = 2 * order - 1;   // 5

    static constexpr Vector<double, order> nodes {
        std::array<double, order> {-0.7745966692414834, 0.000000000000000, 0.7745966692414834}
    };
    static constexpr Vector<double, order> weights {
        std::array<double, order> { 0.5555555555555556, 0.8888888888888888, 0.5555555555555556}
    };
};

// 2D 1 point formula
template <> struct iso_quadrature_gauss_legendre<2, 1> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 2;
    static constexpr int order = 1;
    static constexpr int degree = 2 * order - 1;   // 1

    static constexpr Matrix<double, order, local_dim> nodes {
        std::array<double, order * local_dim> {
            0.000000000000000, 0.000000000000000
        }
    };
    static constexpr Vector<double, order> weights {
        std::array<double, order> {4.000000000000000}
    };
};

// 2D 4 point formula
template <> struct iso_quadrature_gauss_legendre<2, 4> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 2;
    static constexpr int order = 4;
    static constexpr int degree = 2 * order - 1;   // 7

    static constexpr Matrix<double, order, local_dim> nodes {
        std::array<double, order * local_dim> {
            -0.5773502691896257, -0.5773502691896257, 
            -0.5773502691896257, 0.5773502691896257,
            0.5773502691896257, -0.5773502691896257,
            0.5773502691896257, 0.5773502691896257
        }
    };
    static constexpr Vector<double, order> weights {
        std::array<double, order> {
            0.9999999999999996, 0.9999999999999996, 0.9999999999999996, 0.9999999999999996
        }
    };
};

// 2D 9 point formula
template <> struct iso_quadrature_gauss_legendre<2, 9> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 2;
    static constexpr int order = 9;
    static constexpr int degree = 2 * order - 1;   // 17

    static constexpr Matrix<double, order, local_dim> nodes {
        std::array<double, order * local_dim> {
            -0.7745966692414835, -0.7745966692414835,
            -0.7745966692414835,  0.0000000000000000,
            -0.7745966692414835,  0.7745966692414835,
            0.0000000000000000, -0.7745966692414835,
            0.0000000000000000,  0.0000000000000000,
            0.0000000000000000,  0.7745966692414835,
            0.7745966692414835, -0.7745966692414835,
            0.7745966692414835,  0.0000000000000000,
            0.7745966692414835,  0.7745966692414835
        }
    };
    static constexpr Vector<double, order> weights {
        std::array<double, order> {
            0.3086419753086420, 0.4938271604938272, 0.3086419753086420, 0.4938271604938272,
            0.7901234567901235, 0.4938271604938272, 0.3086419753086420, 0.4938271604938272,
            0.3086419753086420
        }
    };
};

// 2D 16 point formula
template <> struct iso_quadrature_gauss_legendre<2, 16> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 2;
    static constexpr int order = 16;
    static constexpr int degree = 2 * 4 - 1;  // 7, since it's a tensor product of 1D 4-point quadrature

    static constexpr std::array<double, 4> one_d_nodes = {
        -0.8611363115940526,
        -0.3399810435848563,
         0.3399810435848563,
         0.8611363115940526
    };

    static constexpr std::array<double, 4> one_d_weights = {
        0.3478548451374538,
        0.6521451548625461,
        0.6521451548625461,
        0.3478548451374538
    };

    static constexpr Matrix<double, order, local_dim> nodes = [] {
        Matrix<double, order, local_dim> m{};
        int k = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m(k, 0) = one_d_nodes[i];  // x
                m(k, 1) = one_d_nodes[j];  // y
                ++k;
            }
        }
        return m;
    }();

    static constexpr Vector<double, order> weights = [] {
        Vector<double, order> w{};
        int k = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                w[k++] = one_d_weights[i] * one_d_weights[j];
        return w;
    }();
};

// 3D 1 point formula
template <> struct iso_quadrature_gauss_legendre<3, 1> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 3;
    static constexpr int order = 1;
    static constexpr int degree = 2 * order - 1;   // 2

    static constexpr Matrix<double, order, local_dim> nodes {
        std::array<double, order * local_dim> {
            0.000000000000000, 0.000000000000000, 0.000000000000000
        }
    };
    static constexpr Vector<double, order> weights {
        std::array<double, order> {8.000000000000000}
    };
};

// 3D 8 point formula
template <> struct iso_quadrature_gauss_legendre<3, 8> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 3;
    static constexpr int order = 8;
    static constexpr int degree = 2 * order - 1;   // 15

static constexpr Matrix<double, order, local_dim> nodes {
    std::array<double, order * local_dim> {
        -0.5773502691896257, -0.5773502691896257, -0.5773502691896257,
            0.5773502691896257, -0.5773502691896257, -0.5773502691896257,
        -0.5773502691896257,  0.5773502691896257, -0.5773502691896257,
            0.5773502691896257,  0.5773502691896257, -0.5773502691896257,
        -0.5773502691896257, -0.5773502691896257,  0.5773502691896257,
            0.5773502691896257, -0.5773502691896257,  0.5773502691896257,
        -0.5773502691896257,  0.5773502691896257,  0.5773502691896257,
            0.5773502691896257,  0.5773502691896257,  0.5773502691896257
    }
};
static constexpr Vector<double, order> weights {
    std::array<double, order> {
        0.9999999999999996, 0.9999999999999996, 0.9999999999999996, 0.9999999999999996,
        0.9999999999999996, 0.9999999999999996, 0.9999999999999996, 0.9999999999999996
    }
};
};

// 3D 27 point formula
template <> struct iso_quadrature_gauss_legendre<3, 27> : public iso_quadrature_gauss_base {
    static constexpr int local_dim = 3;
    static constexpr int order = 27;  // Maintaining order as 8
    static constexpr int degree = 2 * order - 1;   // 15

    static constexpr Matrix<double, order, local_dim> nodes {
        std::array<double, 27 * local_dim> {
            -0.774596669241483, -0.774596669241483, -0.774596669241483,
            -0.774596669241483, -0.774596669241483,  0.000000000000000,
            -0.774596669241483, -0.774596669241483,  0.774596669241483,
            -0.774596669241483,  0.000000000000000, -0.774596669241483,
            -0.774596669241483,  0.000000000000000,  0.000000000000000,
            -0.774596669241483,  0.000000000000000,  0.774596669241483,
            -0.774596669241483,  0.774596669241483, -0.774596669241483,
            -0.774596669241483,  0.774596669241483,  0.000000000000000,
            -0.774596669241483,  0.774596669241483,  0.774596669241483,
             0.000000000000000, -0.774596669241483, -0.774596669241483,
             0.000000000000000, -0.774596669241483,  0.000000000000000,
             0.000000000000000, -0.774596669241483,  0.774596669241483,
             0.000000000000000,  0.000000000000000, -0.774596669241483,
             0.000000000000000,  0.000000000000000,  0.000000000000000,
             0.000000000000000,  0.000000000000000,  0.774596669241483,
             0.000000000000000,  0.774596669241483, -0.774596669241483,
             0.000000000000000,  0.774596669241483,  0.000000000000000,
             0.000000000000000,  0.774596669241483,  0.774596669241483,
             0.774596669241483, -0.774596669241483, -0.774596669241483,
             0.774596669241483, -0.774596669241483,  0.000000000000000,
             0.774596669241483, -0.774596669241483,  0.774596669241483,
             0.774596669241483,  0.000000000000000, -0.774596669241483,
             0.774596669241483,  0.000000000000000,  0.000000000000000,
             0.774596669241483,  0.000000000000000,  0.774596669241483,
             0.774596669241483,  0.774596669241483, -0.774596669241483,
             0.774596669241483,  0.774596669241483,  0.000000000000000,
             0.774596669241483,  0.774596669241483,  0.774596669241483
        }
    };

    static constexpr Vector<double, order> weights {
        std::array<double, order> {
            0.171467764060357, 0.274348422496571, 0.171467764060357, 
            0.274348422496571, 0.438957475994513, 0.274348422496571, 
            0.171467764060357, 0.274348422496571, 0.171467764060357, 
            0.274348422496571, 0.438957475994513, 0.274348422496571, 
            0.438957475994513, 0.702331961591221, 0.438957475994513, 
            0.274348422496571, 0.438957475994513, 0.274348422496571, 
            0.171467764060357, 0.274348422496571, 0.171467764060357, 
            0.274348422496571, 0.438957475994513, 0.274348422496571, 
            0.171467764060357, 0.274348422496571, 0.171467764060357
        }
    };
};

template <int M, typename T>
    requires(requires(T t, int i, int j) {
        { t(i, j) } -> std::same_as<double&>;
        { t.resize(i, j) } -> std::same_as<void>;
    })
void get_iso_quadrature(int degree, T& quad_nodes, T& quad_weights) {

    // Compute number of 1D Gauss points needed per dimension to integrate polynomials of given degree
    int points_per_dim = (degree + 1) / 2 + (degree + 1) % 2;

    // Helper: copy nodes and weights from predefined rule
    auto copy_ = []<typename QuadRule>(const QuadRule& q, T& quad_nodes_, T& quad_weights_) {
        quad_nodes_.resize(q.order, q.local_dim);
        quad_weights_.resize(q.order, 1);
        for (int i = 0; i < q.order; ++i) {
            for (int j = 0; j < q.local_dim; ++j) {
                quad_nodes_(i, j) = q.nodes(i, j);
            }
            quad_weights_(i, 0) = q.weights[i];
        }
    };

    // 1D
    if constexpr (M == 1) {
        if (points_per_dim <= 1) copy_(iso_quadrature_gauss_legendre<1, 1>{}, quad_nodes, quad_weights);
        else if (points_per_dim == 2) copy_(iso_quadrature_gauss_legendre<1, 2>{}, quad_nodes, quad_weights);
        else copy_(iso_quadrature_gauss_legendre<1, 3>{}, quad_nodes, quad_weights);  // safe fallback
    }

    // 2D
    else if constexpr (M == 2) {
        if (points_per_dim <= 1) copy_(iso_quadrature_gauss_legendre<2, 1>{}, quad_nodes, quad_weights);         // 1x1
        else if (points_per_dim == 2) copy_(iso_quadrature_gauss_legendre<2, 4>{}, quad_nodes, quad_weights);    // 2x2
        else if (points_per_dim == 3) copy_(iso_quadrature_gauss_legendre<2, 9>{}, quad_nodes, quad_weights);    // 3x3
        else copy_(iso_quadrature_gauss_legendre<2, 16>{}, quad_nodes, quad_weights);                            // 4x4
    }

    // 3D
    else if constexpr (M == 3) {
        if (points_per_dim <= 1) copy_(iso_quadrature_gauss_legendre<3, 1>{}, quad_nodes, quad_weights);         // 1x1x1
        else if (points_per_dim == 2) copy_(iso_quadrature_gauss_legendre<3, 8>{}, quad_nodes, quad_weights);    // 2x2x2
        else copy_(iso_quadrature_gauss_legendre<3, 27>{}, quad_nodes, quad_weights);                            // 3x3x3
    }
    
}


}// namespace internals


#ifndef __FDAPDE_SP_INTEGRATION_H__
// 1D formulas ( da chiedere se usare quella di spline, altrimenti ci sono conflitti)
[[maybe_unused]] static struct QGL1DP1_ : internals::iso_quadrature_gauss_legendre<1, 1> { } QGL1DP1;
[[maybe_unused]] static struct QGL1DP2_ : internals::iso_quadrature_gauss_legendre<1, 2> { } QGL1DP2;
[[maybe_unused]] static struct QGL1DP3_ : internals::iso_quadrature_gauss_legendre<1, 3> { } QGL1DP3;
#endif
// 2D formulas
[[maybe_unused]] static struct QGL2DP1_ : internals::iso_quadrature_gauss_legendre<2, 1> { } QGL2DP1;
[[maybe_unused]] static struct QGL2DP4_ : internals::iso_quadrature_gauss_legendre<2, 4> { } QGL2DP4;
[[maybe_unused]] static struct QGL2DP9_ : internals::iso_quadrature_gauss_legendre<2, 9> { } QGL2DP9;
[[maybe_unused]] static struct QGL2DP16_ : internals::iso_quadrature_gauss_legendre<2, 16> { } QGL2DP16;
// 3D formulas
[[maybe_unused]] static struct QGL3DP1_ : internals::iso_quadrature_gauss_legendre<3, 1> { } QGL3DP1;
[[maybe_unused]] static struct QGL3DP8_ : internals::iso_quadrature_gauss_legendre<3, 8> { } QGL3DP8;
[[maybe_unused]] static struct QGL3DP27_ : internals::iso_quadrature_gauss_legendre<3, 27> { } QGL3DP27;

}


#endif // __ISO_INTEGRATION_H__