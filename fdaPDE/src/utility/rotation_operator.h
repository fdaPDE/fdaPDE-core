// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_ROTATION_OP_H__
#define __FDAPDE_ROTATION_OP_H__

#include "header_check.h"
#include <cmath>  // for std::sin, std::cos

namespace fdapde {

// forward declaration to break circular dependency
template <typename Scalar, int N> struct RotationOp;

// has_identity trait
namespace internals {

// RotationOp<N> => has identity
template <typename Scalar, int N>
struct has_identity<RotationOp<Scalar, N>> : std::true_type {};

}


// primary template (only enabled for N = 2 or 3 via static_assert)
template <typename Scalar_, int N_>
struct RotationOp : public SquareMatrixBase<N_, RotationOp<Scalar_, N_>> {
    using Base = SquareMatrixBase<N_, RotationOp<Scalar_, N_>>;
    using XprType = RotationOp<Scalar_, N_>;
    using Scalar = Scalar_;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr bool NestAsRefBit = false;
    static constexpr bool ReadOnly = true;
    static constexpr int  XprBits = int(matrix_flags::square) | int(matrix_flags::orthogonal);

    // Constrain allowed sizes
    fdapde_static_assert(N == 2 || N == 3, "RotationOp is only defined for N = 2 or N = 3.");

    // Stub interface; full behavior provided by specializations.
    RotationOp() = default;

    // left multiplication by rotation matrix (generic N)
    template <int RhsRows, int RhsCols, typename RhsType>
    Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& rhs) const {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
        using RScalar = typename RhsType::Scalar;
        Matrix<RScalar, Rows, RhsCols> out;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < RhsCols; ++j) {
                RScalar acc = RScalar(0);
                for (int k = 0; k < Cols; ++k) acc += this->operator()(i, k) * rhs.derived().operator()(k, j);
                out(i, j) = acc;
            }
        }
        return out;
    }

    // right multiplication by rotation matrix (generic N)
    template <int LhsRows, int LhsCols, typename LhsType>
    friend Matrix<typename LhsType::Scalar, LhsRows, Cols>
    operator*(const MatrixBase<LhsRows, LhsCols, LhsType>& lhs, const RotationOp<Scalar, N>& rhs) {
        fdapde_static_assert(LhsCols == Rows, INVALID_OPERANDS_DIMENSION_FOR_MATRIX_MATRIX_PRODUCT);
        using LScalar = typename LhsType::Scalar;
        Matrix<LScalar, LhsRows, Cols> out;
        for (int i = 0; i < LhsRows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                LScalar acc = LScalar(0);
                for (int k = 0; k < LhsCols; ++k) acc += lhs.derived().operator()(i, k) * rhs.operator()(k, j);
                out(i, j) = acc;
            }
        }
        return out;
    }

    // convert to full matrix
    OrthogonalMatrix<Scalar, N> as_matrix() const {
        Matrix<Scalar, N, N> R;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                R(i, j) = this->operator()(i, j);
        return OrthogonalMatrix<Scalar, N>(R);
    }

    // element access — provided by specializations
    Scalar operator()(int, int) const;
};


// specialization for N = 2: rotation in R^2 by angle theta (counterclockwise).
template <typename Scalar_>
struct RotationOp<Scalar_, 2> : public SquareMatrixBase<2, RotationOp<Scalar_, 2>> {
    using Base = SquareMatrixBase<2, RotationOp<Scalar_, 2>>;
    using XprType = RotationOp<Scalar_, 2>;
    using Scalar = Scalar_;
    static constexpr int N = 2;
    static constexpr int Rows = 2;
    static constexpr int Cols = 2;
    static constexpr bool NestAsRefBit = false;
    static constexpr bool ReadOnly = true;
    static constexpr int  XprBits = int(matrix_flags::square) | int(matrix_flags::orthogonal);

    // constructors
    RotationOp() : c_(1.0), s_(0.0) {}
    explicit RotationOp(double theta) : c_(std::cos(theta)), s_(std::sin(theta)) {}

    // left multiplication by rotation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& rhs) const {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
        using RScalar = typename RhsType::Scalar;
        Matrix<RScalar, Rows, RhsCols> out;
        for (int j = 0; j < RhsCols; ++j) {
            // manual unroll for 2x2
            const RScalar r0 = rhs.derived().operator()(0, j);
            const RScalar r1 = rhs.derived().operator()(1, j);
            out(0, j) =  c_ * r0 - s_ * r1;
            out(1, j) =  s_ * r0 + c_ * r1;
        }
        return out;
    }

    // right multiplication by rotation matrix
    template <int LhsRows, int LhsCols, typename LhsType>
    friend Matrix<typename LhsType::Scalar, LhsRows, Cols>
    operator*(const MatrixBase<LhsRows, LhsCols, LhsType>& lhs, const RotationOp& rhs) {
        fdapde_static_assert(LhsCols == Rows, INVALID_OPERANDS_DIMENSION_FOR_MATRIX_MATRIX_PRODUCT);
        using LScalar = typename LhsType::Scalar;
        Matrix<LScalar, LhsRows, Cols> out;
        for (int i = 0; i < LhsRows; ++i) {
            const LScalar l0 = lhs.derived().operator()(i, 0);
            const LScalar l1 = lhs.derived().operator()(i, 1);
            out(i, 0) =  l0 * rhs.c_ + l1 * rhs.s_;
            out(i, 1) = -l0 * rhs.s_ + l1 * rhs.c_;
        }
        return out;
    }

    // const access
    Scalar operator()(int i, int j) const {
        // [ [ c, -s ],
        //   [ s,  c ] ]
        if (i == 0 && j == 0) return c_;
        if (i == 0 && j == 1) return -s_;
        if (i == 1 && j == 0) return s_;
        return c_; // (1,1)
    }

    // convert to full matrix
    OrthogonalMatrix<Scalar, N> as_matrix() const {
        Matrix<Scalar, N, N> R;
        R(0,0) = c_;   R(0,1) = -s_;
        R(1,0) = s_;   R(1,1) =  c_;
        return OrthogonalMatrix<Scalar, N>(R);
    }

private:
    double c_, s_;
};

// specialization for N = 3
//   Rotation in R^3 using yaw-pitch-roll angles:
//     alpha = yaw   (rotation about z-axis)
//     beta  = pitch (rotation about y-axis)
//     gamma = roll  (rotation about x-axis)
//   Composition: R = Rz(alpha) * Ry(beta) * Rx(gamma)
template <typename Scalar_>
struct RotationOp<Scalar_, 3> : public SquareMatrixBase<3, RotationOp<Scalar_, 3>> {
    using Base = SquareMatrixBase<3, RotationOp<Scalar_, 3>>;
    using XprType = RotationOp<Scalar_, 3>;
    using Scalar = Scalar_;
    static constexpr int N = 3;
    static constexpr int Rows = 3;
    static constexpr int Cols = 3;
    static constexpr bool NestAsRefBit = false;
    static constexpr bool ReadOnly = true;
    static constexpr int  XprBits = int(matrix_flags::square) | int(matrix_flags::orthogonal);

    // constructors
    RotationOp()
    : ca_(1.0), sa_(0.0), cb_(1.0), sb_(0.0), cg_(1.0), sg_(0.0) {}

    RotationOp(double alpha, double beta, double gamma)
    : ca_(std::cos(alpha)), sa_(std::sin(alpha)),
      cb_(std::cos(beta)),  sb_(std::sin(beta)),
      cg_(std::cos(gamma)), sg_(std::sin(gamma)) {}

    // Named constructors for axis-aligned rotations
    static RotationOp Rx(double gamma) { return RotationOp(0.0, 0.0, gamma); }
    static RotationOp Ry(double beta)  { return RotationOp(0.0, beta,  0.0 ); }
    static RotationOp Rz(double alpha) { return RotationOp(alpha, 0.0, 0.0 ); }

    // left multiplication by rotation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& rhs) const {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
        using RScalar = typename RhsType::Scalar;
        Matrix<RScalar, Rows, RhsCols> out;

        // Precompute rows of R for speed
        const double r00 =  ca_ * cb_;
        const double r01 =  ca_ * sb_ * sg_ - sa_ * cg_;
        const double r02 =  ca_ * sb_ * cg_ + sa_ * sg_;
        const double r10 =  sa_ * cb_;
        const double r11 =  sa_ * sb_ * sg_ + ca_ * cg_;
        const double r12 =  sa_ * sb_ * cg_ - ca_ * sg_;
        const double r20 = -sb_;
        const double r21 =  cb_ * sg_;
        const double r22 =  cb_ * cg_;

        for (int j = 0; j < RhsCols; ++j) {
            const RScalar x0 = rhs.derived().operator()(0, j);
            const RScalar x1 = rhs.derived().operator()(1, j);
            const RScalar x2 = rhs.derived().operator()(2, j);
            out(0, j) = r00 * x0 + r01 * x1 + r02 * x2;
            out(1, j) = r10 * x0 + r11 * x1 + r12 * x2;
            out(2, j) = r20 * x0 + r21 * x1 + r22 * x2;
        }
        return out;
    }

    // right multiplication by rotation matrix
    template <int LhsRows, int LhsCols, typename LhsType>
    friend Matrix<typename LhsType::Scalar, LhsRows, Cols>
    operator*(const MatrixBase<LhsRows, LhsCols, LhsType>& lhs, const RotationOp& rhs) {
        fdapde_static_assert(LhsCols == Rows, INVALID_OPERANDS_DIMENSION_FOR_MATRIX_MATRIX_PRODUCT);
        using LScalar = typename LhsType::Scalar;

        // Precompute columns of R for speed
        const double r00 =  rhs.ca_ * rhs.cb_;
        const double r01 =  rhs.ca_ * rhs.sb_ * rhs.sg_ - rhs.sa_ * rhs.cg_;
        const double r02 =  rhs.ca_ * rhs.sb_ * rhs.cg_ + rhs.sa_ * rhs.sg_;
        const double r10 =  rhs.sa_ * rhs.cb_;
        const double r11 =  rhs.sa_ * rhs.sb_ * rhs.sg_ + rhs.ca_ * rhs.cg_;
        const double r12 =  rhs.sa_ * rhs.sb_ * rhs.cg_ - rhs.ca_ * rhs.sg_;
        const double r20 = -rhs.sb_;
        const double r21 =  rhs.cb_ * rhs.sg_;
        const double r22 =  rhs.cb_ * rhs.cg_;

        Matrix<LScalar, LhsRows, Cols> out;
        for (int i = 0; i < LhsRows; ++i) {
            const LScalar a0 = lhs.derived().operator()(i, 0);
            const LScalar a1 = lhs.derived().operator()(i, 1);
            const LScalar a2 = lhs.derived().operator()(i, 2);
            out(i, 0) = a0 * r00 + a1 * r10 + a2 * r20;
            out(i, 1) = a0 * r01 + a1 * r11 + a2 * r21;
            out(i, 2) = a0 * r02 + a1 * r12 + a2 * r22;
        }
        return out;
    }

    // const access
    Scalar operator()(int i, int j) const {
        // R = Rz(alpha) * Ry(beta) * Rx(gamma)
        // Entries expanded:
        // [ cα cβ,           cα sβ sγ - sα cγ,   cα sβ cγ + sα sγ ]
        // [ sα cβ,           sα sβ sγ + cα cγ,   sα sβ cγ - cα sγ ]
        // [   -sβ,                 cβ sγ,               cβ cγ     ]
        switch (i) {
            case 0:
                switch (j) {
                    case 0: return  ca_ * cb_;
                    case 1: return  ca_ * sb_ * sg_ - sa_ * cg_;
                    default:return  ca_ * sb_ * cg_ + sa_ * sg_;
                }
            case 1:
                switch (j) {
                    case 0: return  sa_ * cb_;
                    case 1: return  sa_ * sb_ * sg_ + ca_ * cg_;
                    default:return  sa_ * sb_ * cg_ - ca_ * sg_;
                }
            default:
                switch (j) {
                    case 0: return -sb_;
                    case 1: return  cb_ * sg_;
                    default:return  cb_ * cg_;
                }
        }
    }

    // convert to full matrix
    OrthogonalMatrix<Scalar, N> as_matrix() const {
        Matrix<Scalar, N, N> R;
        R(0,0) =  ca_ * cb_;
        R(0,1) =  ca_ * sb_ * sg_ - sa_ * cg_;
        R(0,2) =  ca_ * sb_ * cg_ + sa_ * sg_;
        R(1,0) =  sa_ * cb_;
        R(1,1) =  sa_ * sb_ * sg_ + ca_ * cg_;
        R(1,2) =  sa_ * sb_ * cg_ - ca_ * sg_;
        R(2,0) = -sb_;
        R(2,1) =  cb_ * sg_;
        R(2,2) =  cb_ * cg_;
        return OrthogonalMatrix<Scalar, N>(R);
    }

private:
    // cached sines/cosines
    double ca_, sa_; // cos/sin(alpha) yaw  (z)
    double cb_, sb_; // cos/sin(beta)  pitch (y)
    double cg_, sg_; // cos/sin(gamma) roll  (x)
};

} // namespace fdapde

#endif // __FDAPDE_ROTATION_OP_H__