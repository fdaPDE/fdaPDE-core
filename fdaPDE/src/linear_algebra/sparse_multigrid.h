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

#ifndef __FDAPDE_SPARSE_MULTIGRID_H__
#define __FDAPDE_SPARSE_MULTIGRID_H__
#include "header_check.h"

namespace fdapde {

/// @brief fixed symmetric smoothed-aggregation cycles for sparse SPD scalar elliptic systems
/// @details solve applies a linear approximate inverse, not a tolerance-stopped solve
class SparseMultigrid {
   public:
    using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Vector = Eigen::VectorXd;

    /// @brief build a Galerkin hierarchy with paired Gauss-Seidel sweeps and an exact coarse solve
    explicit SparseMultigrid(const Eigen::SparseMatrix<double>& matrix, int cycles = 4, int coarse_size = 256) :
        cycles_(cycles) {
        fdapde_strong_assert(
          matrix.rows() > 0 && matrix.rows() == matrix.cols(), std::invalid_argument,
          "multigrid requires a nonempty square matrix");
        fdapde_strong_assert(
          cycles > 0 && cycles <= 16 && coarse_size >= 2, std::invalid_argument,
          "invalid multigrid cycle count or coarse size");
        for (int j = 0; j < matrix.outerSize(); ++j)
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, j); it; ++it)
                fdapde_strong_assert(std::isfinite(it.value()), std::invalid_argument, "nonfinite multigrid matrix");
        const Eigen::SparseMatrix<double> difference = matrix - Eigen::SparseMatrix<double>(matrix.transpose());
        fdapde_strong_assert(
          difference.norm() <= 1e-12 * matrix.norm(), std::invalid_argument, "multigrid requires a symmetric matrix");
        matrices_.push_back(matrix);
        while (matrices_.back().rows() > coarse_size) {
            const Sparse& a = matrices_.back();
            const Vector diagonal = a.diagonal();
            fdapde_strong_assert(
              (diagonal.array() > 0).all(), std::domain_error, "multigrid requires positive diagonal entries");
            const auto aggregate = aggregates_(a, diagonal);
            const int count = *std::max_element(aggregate.begin(), aggregate.end()) + 1;
            if (count == a.rows()) break;
            fdapde_strong_assert(matrices_.size() < 20, std::runtime_error, "multigrid hierarchy did not coarsen");
            double bound = 0.;
            for (int i = 0; i < a.rows(); ++i) {
                double sum = 0.;
                for (Sparse::InnerIterator it(a, i); it; ++it) sum += std::abs(it.value());
                bound = std::max(bound, sum / diagonal[i]);
            }
            // a subcritical Jacobi step keeps the transfer full rank
            const double omega = .8 / bound;
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(a.nonZeros() + a.rows());
            for (int i = 0; i < a.rows(); ++i) {
                triplets.emplace_back(i, aggregate[i], 1.);
                for (Sparse::InnerIterator it(a, i); it; ++it)
                    triplets.emplace_back(i, aggregate[it.col()], -omega * it.value() / diagonal[i]);
            }
            Sparse p(a.rows(), count);
            p.setFromTriplets(triplets.begin(), triplets.end());
            p.prune(0.);
            Sparse next = p.transpose() * (a * p);
            next = (.5 * (next + Sparse(next.transpose()))).eval();
            next.prune(0.);
            transfers_.push_back(std::move(p));
            matrices_.push_back(std::move(next));
        }
        coarse_.compute(Eigen::SparseMatrix<double>(matrices_.back()));
        fdapde_strong_assert(
          coarse_.info() == Eigen::Success, std::domain_error, "multigrid coarse matrix is not positive definite");
    }

    /// @brief apply a fixed number of symmetric cycles from zero with call-local workspaces
    Vector solve(const Vector& rhs) const {
        fdapde_strong_assert(
          rhs.size() == matrices_.front().rows() && rhs.allFinite(), std::invalid_argument,
          "invalid multigrid right hand side");
        Vector x = Vector::Zero(rhs.size());
        for (int i = 0; i < cycles_; ++i) x += cycle_(0, rhs - matrices_.front() * x);
        return x;
    }

    /// @brief expose hierarchy sizes for reproducible complexity diagnostics
    std::vector<int> level_sizes() const {
        std::vector<int> sizes;
        for (const auto& a : matrices_) sizes.push_back(a.rows());
        return sizes;
    }

    /// @brief count stored Galerkin and transfer coefficients excluding coarse-factor fill
    Eigen::Index nonzeros() const {
        Eigen::Index count = 0;
        for (const auto& a : matrices_) count += a.nonZeros();
        for (const auto& p : transfers_) count += p.nonZeros();
        return count;
    }
   private:
    std::vector<Sparse> matrices_, transfers_;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> coarse_;
    int cycles_;

    /// @brief group unassigned strongly connected neighbors deterministically
    static std::vector<int> aggregates_(const Sparse& a, const Vector& diagonal) {
        std::vector<int> ids(a.rows(), -1);
        int count = 0;
        for (int i = 0; i < a.rows(); ++i) {
            if (ids[i] >= 0) continue;
            ids[i] = count;
            double strongest = 0.;
            for (Sparse::InnerIterator it(a, i); it; ++it)
                if (it.col() != i)
                    strongest = std::max(strongest, std::abs(it.value()) / std::sqrt(diagonal[i] * diagonal[it.col()]));
            for (Sparse::InnerIterator it(a, i); it; ++it)
                if (
                  ids[it.col()] < 0 && std::abs(it.value()) > 0 &&
                  std::abs(it.value()) / std::sqrt(diagonal[i] * diagonal[it.col()]) >= .25 * strongest)
                    ids[it.col()] = count;
            ++count;
        }
        // attach stranded singletons to established neighbors to avoid stalled coarse levels
        std::vector<int> sizes(count, 0);
        for (int id : ids) ++sizes[id];
        for (int i = 0; i < a.rows(); ++i) {
            if (sizes[ids[i]] != 1) continue;
            int target = ids[i];
            double strongest = 0.;
            for (Sparse::InnerIterator it(a, i); it; ++it) {
                if (sizes[ids[it.col()]] < 2) continue;
                const double strength = std::abs(it.value()) / std::sqrt(diagonal[i] * diagonal[it.col()]);
                if (strength > strongest) {
                    strongest = strength;
                    target = ids[it.col()];
                }
            }
            if (target != ids[i]) {
                --sizes[ids[i]];
                ++sizes[target];
                ids[i] = target;
            }
        }
        std::vector<int> compact(count, -1);
        count = 0;
        for (int& id : ids) {
            if (compact[id] < 0) compact[id] = count++;
            id = compact[id];
        }
        return ids;
    }

    /// @brief pair forward and backward sweeps around the recursive coarse correction
    Vector cycle_(std::size_t level, const Vector& rhs) const {
        if (level + 1 == matrices_.size()) return coarse_.solve(rhs);
        const Sparse& a = matrices_[level];
        const Sparse& p = transfers_[level];
        Vector x = Vector::Zero(rhs.size());
        for (int sweep = 0; sweep < 2; ++sweep)
            x += a.template triangularView<Eigen::Lower>().solve(Vector(rhs - a * x));
        x += p * cycle_(level + 1, p.transpose() * (rhs - a * x));
        for (int sweep = 0; sweep < 2; ++sweep)
            x += a.template triangularView<Eigen::Upper>().solve(Vector(rhs - a * x));
        return x;
    }
};
}   // namespace fdapde
#endif   // __FDAPDE_SPARSE_MULTIGRID_H__
