#ifndef __INTEGRATION_TABLES_H__
#define __INTEGRATION_TABLES_H__

#include <vector>
#include <array>
#include <cmath>


namespace fdapde {

    // mettere namespace internalss ?? 

/**
 * @brief Integration table for Lobatto-Gauss-Legendre (LGL) quadrature rules.
 * @tparam N Dimension of the integration domain.
 * @tparam K Number of quadrature nodes.
 */
template<int N, int K> struct IntegratorTableLGL;

// 1D line elements
// 3 point formula, Gauss-Legendre rule on interval [-1,1]
template <>
struct IntegratorTableLGL<1, 2> {
    static constexpr int local_dim = 1;
    static constexpr int order = 3;

    static constexpr cexpr::Vector<double, order> nodes { 
        std::array<double, order> {-0.774596669241483, 0.000000000000000,0.774596669241483 }
        };
    static constexpr cexpr::Vector<double, order> weights { 
        std::array<double, order> {0.555555555555555, 0.888888888888888, 0.555555555555555}
        };
};

// 2D quad elements
// reference element: square made from cartesian product (-1,1)x(-1,1) 

// 4 point formula, degree of precision 3
template <>
struct IntegratorTableLGL<2, 4> {
    static constexpr int local_dim = 2;
    static constexpr int order = 4;

    static constexpr cexpr::Matrix<double, order, local_dim> nodes {
        std::array<double, order * local_dim> {
            -0.5773502691896257, -0.5773502691896257, 
            -0.5773502691896257, 0.5773502691896257,
            0.5773502691896257, -0.5773502691896257,
            0.5773502691896257, 0.5773502691896257
        }
    };
    static constexpr cexpr::Vector<double, order> weights {
        std::array<double, order> {
            0.9999999999999996, 0.9999999999999996, 0.9999999999999996, 0.9999999999999996
        }
    };
};

// 9 point formula, degree of precision 5
template<>
struct IntegratorTableLGL<2, 9> {
    static constexpr int local_dim = 2;
    static constexpr int order = 9;

    static constexpr cexpr::Matrix<double, order, local_dim> nodes {
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
    static constexpr cexpr::Vector<double, order> weights {
        std::array<double, order> {
            0.3086419753086420, 0.4938271604938272, 0.3086419753086420, 0.4938271604938272,
            0.7901234567901235, 0.4938271604938272, 0.3086419753086420, 0.4938271604938272,
            0.3086419753086420
        }
    };
};

// 3D cube elements
// 27 point formula, degree of precision 3
template<>
struct IntegratorTableLGL<3, 27> {
    static constexpr int local_dim = 3;
    static constexpr int order = 27;

    // Nodes for the 27-point quadrature rule
    static constexpr cexpr::Matrix<double, order, local_dim> nodes {
        std::array<double, order * local_dim> {
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

    // Weights for the 27-point quadrature rule
    static constexpr cexpr::Vector<double, order> weights {
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


} // namespace fdapde

#endif // __INTEGRATION_TABLES_H__