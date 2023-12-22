#ifndef __INTEGRATOR_TABLES_H__
#define __INTEGRATOR_TABLES_H__

#include <array>

namespace fdapde {
namespace core {

// Collection of weights and nodes for quadrature formulae to be applied on a reference mesh element
// (N-dimensional unit simplex).
// Tables and theory for the derivation of both weights and nodes can be found in:
// ** "Numerical Models for Differential Problems, Alfio Quarteroni. Second edition"
// ** "https://people.sc.fsu.edu/~jburkardt/datasets/datasets.html"

// N: dimension of the integration domain, K: number of nodes of the formula. Args: type of quadrature formula
template <int N, int K, typename... Args> struct IntegratorTable;
// identification tags for quadrature rules
struct NewtonCotes { };
struct GaussLegendre { };

// trait for selecting a standard quadrature rule for finite elements, the rules are exact for polynomials of degree
// fem_order + 1, since we might integrate products of order fe_order polynomials (e.g. weak form of a reaction term)
template <int N, int R>
struct standard_fem_quadrature_rule {
    static constexpr int quadrature(const int dim, const int fem_order) { 
        switch (dim) {
        case 1:   // 1D elements
            switch (fem_order) {
            case 1:         // linear elements
                return 2;   // 2 point rule
            case 2:         // quadratic elements
                return 3;   // 3 point rule
            default:
                return 3;
            }
        case 2:   // 2D elements
            switch (fem_order) {
            case 1:         // linear elements
                return 3;   // 3 point rule
            case 2:         // quadratic elements
                return 6;   // 6 point rule
            default:
                return 12;
            }
        case 3:   // 3D elements
            switch (fem_order) {
            case 1:         // linear elements
                return 4;   // 4 point rule
            case 2:         // quadratic elements
                return 5;   // 5 point rule
            default:
                return 5;
            }
        }
        return 0;   // error
    }
    static constexpr int K = quadrature(N, R);
};

// 1D linear elements (gaussian integration)
// reference element: simplex of vertices (0), (1)

// 2 point formula
template <> struct IntegratorTable<1, 2> {
    enum {
        input_dim = 1, // input space dimension of integrand field
        num_nodes = 2  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<1>, 2> nodes = {
       SVector<1>(0.211324865405187), SVector<1>(0.788675134594812)
    };
    // weights of the quadrature rule
    std::array<double, 2> weights = {
       0.500000000000000, 0.500000000000000
    };
};

// 3 point formula
template <> struct IntegratorTable<1, 3> {
    enum {
        input_dim = 1, // input space dimension of integrand field
        num_nodes = 3  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<1>, 3> nodes = {
       SVector<1>(0.112701665379258), SVector<1>(0.500000000000000), SVector<1>(0.887298334620741)
    };
    // weights of the quadrature rule
    std::array<double, 3> weights = {
       0.277777777777778, 0.444444444444444, 0.277777777777778
    };
};

// 3 point formula, Gauss-Legendre rule on interval [-1,1]
template <> struct IntegratorTable<1, 3, GaussLegendre> {
    enum {
        input_dim = 1, // input space dimension of integrand field
        num_nodes = 3  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<1>, 3> nodes = {
       SVector<1>(-0.774596669241483), SVector<1>(0.000000000000000), SVector<1>(0.774596669241483)
    };
    // weights of the quadrature rule
    std::array<double, 3> weights = {
       0.555555555555555, 0.888888888888888, 0.555555555555555
    };
};

// 2D triangular elements
// reference element: simplex of vertices (0,0), (1,0), (0,1)

// 1 point formula
template <> struct IntegratorTable<2, 1> {
    enum {
        input_dim = 2, // input space dimension of integrand field
        num_nodes = 1  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<2>, 1> nodes = {
       SVector<2>(0.333333333333333, 0.333333333333333)
    };
    // weights of the quadrature rule
    std::array<double, 1> weights = {1.};
};

// 3 point formula
template <> struct IntegratorTable<2, 3> {
    enum {
        input_dim = 2, // input space dimension of integrand field
        num_nodes = 3  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<2>, 3> nodes = {
       SVector<2>(0.166666666666667, 0.166666666666667),
       SVector<2>(0.666666666666667, 0.166666666666667),
       SVector<2>(0.166666666666667, 0.666666666666667)
    };
    // weights of the quadrature rule
    std::array<double, 3> weights = {
       0.333333333333333, 0.333333333333333, 0.333333333333333
    };
};

// 6 point formula
template <> struct IntegratorTable<2, 6> {
    enum {
        input_dim = 2, // input space dimension of integrand field
        num_nodes = 6  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<2>, 6> nodes = {
       SVector<2>(0.445948490915965, 0.445948490915965),
       SVector<2>(0.445948490915965, 0.108103018168070),
       SVector<2>(0.108103018168070, 0.445948490915965),
       SVector<2>(0.091576213509771, 0.091576213509771),
       SVector<2>(0.091576213509771, 0.816847572980459),
       SVector<2>(0.816847572980459, 0.091576213509771)
    };
    // weights of the quadrature rule
    std::array<double, 6> weights = {
       0.223381589678011, 0.223381589678011, 0.223381589678011,
       0.109951743655322, 0.109951743655322, 0.109951743655322
    };
};

// 7 point formula
template <> struct IntegratorTable<2, 7> {
    enum {
        input_dim = 2, // input space dimension of integrand field
        num_nodes = 7  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<2>, 7> nodes = {
       SVector<2>(0.333333333333333, 0.333333333333333),
       SVector<2>(0.101286507323456, 0.101286507323456),
       SVector<2>(0.101286507323456, 0.797426985353087),
       SVector<2>(0.797426985353087, 0.101286507323456),
       SVector<2>(0.470142064105115, 0.470142064105115),
       SVector<2>(0.470142064105115, 0.059715871789770),
       SVector<2>(0.059715871789770, 0.470142064105115)
    };
    // weights of the quadrature rule
    std::array<double, 7> weights = {
       0.225000000000000, 0.125939180544827, 0.125939180544827,
       0.125939180544827, 0.132394152788506, 0.132394152788506,
       0.132394152788506
    };
};

// 12 point formula, degree of precision 6
template <> struct IntegratorTable<2, 12> {
    enum {
        input_dim = 2,  // input space dimension of integrand field
        num_nodes = 12  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<2>, 12> nodes = {
      SVector<2>(0.873821971016996, 0.063089014491502),
      SVector<2>(0.063089014491502, 0.873821971016996),
      SVector<2>(0.063089014491502, 0.063089014491502),
      SVector<2>(0.501426509658179, 0.249286745170910),
      SVector<2>(0.249286745170910, 0.501426509658179),
      SVector<2>(0.249286745170910, 0.249286745170910),
      SVector<2>(0.636502499121399, 0.310352451033785),
      SVector<2>(0.636502499121399, 0.053145049844816),
      SVector<2>(0.310352451033785, 0.636502499121399),
      SVector<2>(0.310352451033785, 0.053145049844816),
      SVector<2>(0.053145049844816, 0.636502499121399),
      SVector<2>(0.053145049844816, 0.310352451033785)
    };
    // weights of the quadrature rule
    std::array<double, 12> weights = {
      0.050844906370207, 0.050844906370207, 0.050844906370207, 0.116786275726379,
      0.116786275726379, 0.116786275726379, 0.082851075618374, 0.082851075618374,
      0.082851075618374, 0.082851075618374, 0.082851075618374, 0.082851075618374
    };
};

// 3D tetrahedric elements
// reference element: simplex of vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)

// 1 point formula
template <> struct IntegratorTable<3, 1> {
    enum {
        input_dim = 3, // input space dimension of integrand field
        num_nodes = 1  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<3>, 1> nodes = {
       SVector<3>(0.250000000000000, 0.250000000000000, 0.250000000000000)
    };
    // weights of the quadrature rule
    std::array<double, 1> weights = {1.};
};

// 4 point formula, degree of precision 2
template <> struct IntegratorTable<3, 4> {
    enum {
        input_dim = 3, // input space dimension of integrand field
        num_nodes = 4  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<3>, 4> nodes = {
       SVector<3>(0.585410196624969, 0.138196601125011, 0.138196601125011),
       SVector<3>(0.138196601125011, 0.138196601125011, 0.138196601125011),
       SVector<3>(0.138196601125011, 0.138196601125011, 0.585410196624969),
       SVector<3>(0.138196601125011, 0.585410196624969, 0.138196601125011)
    };
    // weights of the quadrature rule
    std::array<double, 4> weights = {
       0.250000000000000, 0.250000000000000, 0.250000000000000, 0.250000000000000
    };
};

// 5 point formula, degree of precision 3
template <> struct IntegratorTable<3, 5> {
    enum {
        input_dim = 3, // input space dimension of integrand field
        num_nodes = 5  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<3>, 5> nodes = {
       SVector<3>(0.250000000000000, 0.250000000000000, 0.250000000000000),
       SVector<3>(0.500000000000000, 0.166666666666667, 0.166666666666667),
       SVector<3>(0.166666666666667, 0.500000000000000, 0.166666666666667),
       SVector<3>(0.166666666666667, 0.166666666666667, 0.500000000000000),
       SVector<3>(0.166666666666667, 0.166666666666667, 0.166666666666667)
    };
    // weights of the quadrature rule
    std::array<double, 5> weights = {
      -0.80000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000
    };
};

// 5 point formula, degree of precision 4
template <> struct IntegratorTable<3, 11> {
    enum {
        input_dim = 3,  // input space dimension of integrand field
        num_nodes = 11  // number of qudrature nodes
    };
    // position of nodes (in barycentric coordinates)
    std::array<SVector<3>, 11> nodes = {
       SVector<3>(0.2500000000000000, 0.2500000000000000, 0.2500000000000000),
       SVector<3>(0.7857142857142857, 0.0714285714285714, 0.0714285714285714),
       SVector<3>(0.0714285714285714, 0.0714285714285714, 0.0714285714285714),
       SVector<3>(0.0714285714285714, 0.0714285714285714, 0.7857142857142857),
       SVector<3>(0.0714285714285714, 0.7857142857142857, 0.0714285714285714),
       SVector<3>(0.1005964238332008, 0.3994035761667992, 0.3994035761667992),
       SVector<3>(0.3994035761667992, 0.1005964238332008, 0.3994035761667992),
       SVector<3>(0.3994035761667992, 0.3994035761667992, 0.1005964238332008),
       SVector<3>(0.3994035761667992, 0.1005964238332008, 0.1005964238332008),
       SVector<3>(0.1005964238332008, 0.3994035761667992, 0.1005964238332008),
       SVector<3>(0.1005964238332008, 0.1005964238332008, 0.3994035761667992)
    };
    // weights of the quadrature rule
    std::array<double, 11> weights = {
      -0.0789333333333333, 0.0457333333333333, 0.0457333333333333, 0.0457333333333333, 0.0457333333333333,
       0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333,
       0.1493333333333333
    };
};

}   // namespace core
}   // namespace fdaPDE

#endif   // __INTEGRATOR_TABLES_H__
