#ifndef __SP_PERIODIC_TRANSFORMATION_H__
#define __SP_PERIODIC_TRANSFORMATION_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

inline Eigen::Matrix<double, 2, 2> periodic_T_p1(int /*e*/, int /*n_el*/) {
    Eigen::Matrix<double, 2, 2> T;
    T << 1.0, 0.0,
         0.0, 1.0;
    return T;
}

inline Eigen::Matrix<double, 3, 3> periodic_T_p2(int e, int n_el) {
    Eigen::Matrix<double, 3, 3> T;
    if (e == 0) {
        T << 0.5, 0.0, 0.0,
             0.5, 1.0, 0.0,
             0.0, 0.0, 1.0;
    } else if (e == n_el-1) {
        T << 1.0, 0.0, 0.0,
             0.0, 1.0, 0.5,
             0.0, 0.0, 0.5;
    } else {
        T.setIdentity();
    }
    return T;
}

inline Eigen::Matrix<double, 4, 4> periodic_T_p3(int e, int n_el) {
    Eigen::Matrix<double, 4, 4> T;
    if (e == 0) {
        T << 1.0/6.0, 0.0,     0.0,     0.0,
             2.0/3.0, 2.0/3.0, 0.0,     0.0,
             1.0/6.0, 1.0/3.0, 1.0,     0.0,
             0.0,     0.0,     0.0,     1.0;
    } else if (e == 1) {
        T << 2.0/3.0, 0.0,     0.0,     0.0,
             1.0/3.0, 1.0,     0.0,     0.0,
             0.0,     0.0,     1.0,     0.0,
             0.0,     0.0,     0.0,     1.0;
    } else if (e == n_el - 2) {
        T << 1.0, 0.0,     0.0,     0.0,
             0.0, 1.0,     0.0,     0.0,
             0.0, 0.0,     1.0,     1.0/3.0,
             0.0, 0.0,     0.0,     2.0/3.0;
    } else if (e == n_el-1) {
        T << 1.0, 0.0,     0.0,     0.0,
             0.0, 1.0,     1.0/3.0, 1.0/6.0,
             0.0, 0.0,     2.0/3.0, 2.0/3.0,
             0.0, 0.0,     0.0,     1.0/6.0;
    } else {
        T.setIdentity();
    }
    return T;
}

// Entry point (runtime degree-aware)
inline Eigen::Matrix<double, Dynamic, Dynamic> bs_periodic_transformation(int p, int e, int n_el)  {
    fdapde_assert(p >= 1 && p <= 3);
    switch (p) {
        case 1: return periodic_T_p1(e, n_el);
        case 2: return periodic_T_p2(e, n_el);
        case 3: return periodic_T_p3(e, n_el);
    }
    return Eigen::Matrix<double, Dynamic, Dynamic>(); // fallback (won’t be reached if assertion holds)
}

} // namespace internals
} // namespace fdapde

#endif // __ISO_PERIODIC_TRANSFORMATION_H__