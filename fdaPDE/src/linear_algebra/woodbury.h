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

#ifndef __FDAPDE_LINALG_WOODBURY_H__
#define __FDAPDE_LINALG_WOODBURY_H__

#include <cmath>
#include <concepts>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "header_check.h"

namespace fdapde {
namespace internals {

template <typename Solver> constexpr Solver& unwrap_woodbury_solver(Solver& solver) { return solver; }
template <typename Solver> constexpr Solver& unwrap_woodbury_solver(std::reference_wrapper<Solver>& solver) {
    return solver.get();
}

template <typename Solver>
concept woodbury_solver = requires(std::remove_cvref_t<Solver>& solver, const Vector<double, Dynamic>& rhs) {
    requires matrix_expression<decltype(unwrap_woodbury_solver(solver).solve(rhs))>;
};

}   // namespace internals

// Solver for (A + U C V)x = b using the Sherman-Morrison-Woodbury
// identity. The public input named inverse_c is C^{-1}, matching the stable
// API. The base solver is owned, A^{-1}U is cached, and all dense state is
// materialized so temporaries and copies are independent. Supplying an
// explicit std::reference_wrapper opts into caller-managed solver lifetime;
// the one-shot compatibility helper below uses that only within its call.
//
// TODO(P4-M): select a concrete native sparse factorization/backend only
// after a dependency-closed consumer and performance evidence require one.
// This adaptor intentionally depends only on a dense-vector solve() contract.
template <typename SparseSolver_>
    requires(internals::woodbury_solver<std::remove_cvref_t<SparseSolver_>>)
class Woodbury {
   public:
    using Scalar = double;
    using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
    using SparseSolver = std::remove_cvref_t<SparseSolver_>;
    using DenseSolver = PartialPivLU<MatrixType>;

    Woodbury() = default;

    template <
      internals::matrix_expression UType, internals::matrix_expression CInvType, internals::matrix_expression VType>
        requires(
          std::convertible_to<typename UType::Scalar, Scalar> &&
          std::convertible_to<typename CInvType::Scalar, Scalar> && std::convertible_to<typename VType::Scalar, Scalar>)
    Woodbury(
      SparseSolver solver, const MatrixExpr<UType>& u, const MatrixExpr<CInvType>& inverse_c,
      const MatrixExpr<VType>& v) {
        initialize_(std::move(solver), u.derived(), inverse_c.derived(), v.derived());
    }

    Woodbury(const Woodbury&) = default;
    Woodbury& operator=(const Woodbury&) = default;
    Woodbury(Woodbury&& other) : state_(std::move(other.state_)) { other.state_.reset(); }
    Woodbury& operator=(Woodbury&& other) {
        if (this != &other) {
            state_ = std::move(other.state_);
            other.state_.reset();
        }
        return *this;
    }

    template <internals::matrix_expression RhsType>
        requires(std::convertible_to<typename RhsType::Scalar, Scalar>)
    MatrixType solve(const MatrixExpr<RhsType>& rhs) {
        if (!state_) { throw std::domain_error("Woodbury solve requires an initialized decomposition"); }
        const RhsType& rhs_derived = rhs.derived();
        if (rhs_derived.rows() != state_->dimension || rhs_derived.cols() <= 0) {
            throw std::invalid_argument("Woodbury requires a matching nonempty right-hand side");
        }
        if (!all_finite_(rhs_derived)) {
            throw std::invalid_argument("Woodbury requires finite right-hand-side coefficients");
        }

        const MatrixType rhs_owned(rhs_derived);
        MatrixType y = base_solve_(state_->solver, rhs_owned, state_->dimension, rhs_owned.cols());
        const MatrixType projected(state_->v * y);
        if (!all_finite_(projected)) { throw std::domain_error("Woodbury projection produced nonfinite values"); }
        const MatrixType t(state_->inverse_g.solve(projected));
        if (t.rows() != state_->rank || t.cols() != rhs_owned.cols() || !all_finite_(t)) {
            throw std::domain_error("Woodbury correction solve produced an invalid result");
        }
        const MatrixType correction(state_->inverse_a_u * t);
        const MatrixType result(y - correction);
        if (!all_finite_(result)) { throw std::domain_error("Woodbury solve produced nonfinite values"); }
        return result;
    }
   private:
    struct State {
        SparseSolver solver;
        MatrixType inverse_a_u;
        MatrixType v;
        DenseSolver inverse_g;
        int dimension;
        int rank;

        State(
          SparseSolver solver_, MatrixType inverse_a_u_, MatrixType v_, DenseSolver inverse_g_, int dimension_,
          int rank_) :
            solver(std::move(solver_)),
            inverse_a_u(std::move(inverse_a_u_)),
            v(std::move(v_)),
            inverse_g(std::move(inverse_g_)),
            dimension(dimension_),
            rank(rank_) { }
    };

    template <typename XprType> static bool all_finite_(const XprType& xpr) {
        for (int row = 0; row < xpr.rows(); ++row) {
            for (int col = 0; col < xpr.cols(); ++col) {
                if (!std::isfinite(static_cast<Scalar>(xpr(row, col)))) return false;
            }
        }
        return true;
    }

    static MatrixType base_solve_(SparseSolver& solver, const MatrixType& rhs, int expected_rows, int expected_cols) {
        MatrixType result(expected_rows, expected_cols);
        for (int col = 0; col < expected_cols; ++col) {
            Vector<Scalar, Dynamic> rhs_column(expected_rows);
            for (int row = 0; row < expected_rows; ++row) rhs_column[row] = rhs(row, col);
            auto solution = internals::unwrap_woodbury_solver(solver).solve(rhs_column);
            if (solution.rows() != expected_rows || solution.cols() != 1 || !all_finite_(solution)) {
                throw std::domain_error("Woodbury base solver produced an invalid result");
            }
            for (int row = 0; row < expected_rows; ++row) result(row, col) = static_cast<Scalar>(solution(row, 0));
        }
        return result;
    }

    template <typename UType, typename CInvType, typename VType>
    void initialize_(SparseSolver solver, const UType& u, const CInvType& inverse_c, const VType& v) {
        const int dimension = u.rows();
        const int rank = u.cols();
        if (
          dimension <= 0 || rank <= 0 || inverse_c.rows() != rank || inverse_c.cols() != rank || v.rows() != rank ||
          v.cols() != dimension) {
            throw std::invalid_argument("Woodbury requires U(n,q), inverse_c(q,q), and V(q,n) with n,q positive");
        }
        if (!all_finite_(u) || !all_finite_(inverse_c) || !all_finite_(v)) {
            throw std::invalid_argument("Woodbury requires finite update coefficients");
        }

        const MatrixType u_owned(u);
        const MatrixType inverse_c_owned(inverse_c);
        MatrixType v_owned(v);
        MatrixType inverse_a_u = base_solve_(solver, u_owned, dimension, rank);
        const MatrixType g(inverse_c_owned + v_owned * inverse_a_u);
        if (!all_finite_(g)) { throw std::domain_error("Woodbury correction matrix contains nonfinite values"); }
        DenseSolver inverse_g(g);
        if (inverse_g.info() != 0) { throw std::domain_error("Woodbury requires a nonsingular correction matrix"); }

        state_.emplace(
          std::move(solver), std::move(inverse_a_u), std::move(v_owned), std::move(inverse_g), dimension, rank);
    }

    std::optional<State> state_;
};

template <typename Solver, typename UType, typename CInvType, typename VType>
Woodbury(Solver, const MatrixExpr<UType>&, const MatrixExpr<CInvType>&, const MatrixExpr<VType>&)
  -> Woodbury<std::remove_cvref_t<Solver>>;

// One-shot compatibility spelling. A reference wrapper avoids copying an
// already-computed backend while the owning Woodbury object remains the
// repeated-solve API.
template <
  typename Solver, internals::matrix_expression UType, internals::matrix_expression CInvType,
  internals::matrix_expression VType, internals::matrix_expression RhsType>
auto woodbury_system_solve(
  Solver&& solver, const MatrixExpr<UType>& u, const MatrixExpr<CInvType>& inverse_c, const MatrixExpr<VType>& v,
  const MatrixExpr<RhsType>& rhs) {
    auto solver_reference = std::ref(solver);
    Woodbury decomposition(solver_reference, u, inverse_c, v);
    return decomposition.solve(rhs);
}

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_WOODBURY_H__
