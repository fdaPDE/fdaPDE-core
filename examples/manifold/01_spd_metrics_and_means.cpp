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

// compile from the repository root with c++ -std=c++20 -O2 -pthread -I.
// examples/manifold/01_spd_metrics_and_means.cpp
#include <fdaPDE/manifold_means.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace fdapde;
using namespace fdapde::manifold;

namespace {
using Policy = Cache::Union<Cache::Spectral, Cache::Log, Cache::Sqrt, Cache::InverseSqrt>;
using SPD = SPDMatrix<double, 2, Policy>;

/// @brief measures full coefficient error against an independently constructed SPD oracle
double error(const SPD& actual, const SPD& expected) { return Matrix<double, 2, 2>(actual - expected).norm(); }

/// @brief prints one metric comparison and its separately computed diagonal barycenter
void report(const char* name, double distance, double squared_norm, const WeightedKarcherMeanResult<SPD>& mean) {
    std::cout << std::setw(14) << name << "  distance=" << distance << "  <log,log>=" << squared_norm << "  mean=diag("
              << mean.point(0, 0) << ", " << mean.point(1, 1) << ")"
              << "  iterations=" << mean.iterations << "  residual=" << mean.stationarity_norm << '\n';
}
}   // namespace

// compares concrete SPD metrics, checks inverse maps and transports, then solves weighted means with scalar oracles
int main() {
    std::cout << std::fixed << std::setprecision(8);
    // the same noncommuting SPD coefficients acquire different distances under each metric
    const SPD p(Vector<double, 3> {2., .3, 1.5});
    const SPD q(Vector<double, 3> {1.4, -.2, 2.2});
    std::cout << "Endpoints: P=[[2,.3],[.3,1.5]], Q=[[1.4,-.2],[-.2,2.2]]\n";

    // diagonal samples provide independent scalar formulas for the normalized mean with weights (1,3)
    MatrixBatch<SPD> samples(2);
    samples[0] = SPD(Vector<double, 3> {1., 0., 4.});
    samples[1] = SPD(Vector<double, 3> {4., 0., 9.});
    const std::array<double, 2> weights {1., 3.};
    const SPD geometric_mean(Vector<double, 3> {std::pow(4., .75), 0., std::pow(4., .25) * std::pow(9., .75)});
    const SPD root_mean(Vector<double, 3> {std::pow(.25 * 1. + .75 * 2., 2), 0., std::pow(.25 * 2. + .75 * 3., 2)});
    const SPD initial = SPD::Identity();
    WeightedKarcherMeanOptions options;
    options.solver.max_iterations = 200;
    options.solver.gradient_tolerance = 1e-10;
    std::cout << "Means: samples diag(1,4), diag(4,9); normalized weights (.25,.75)\n";

    // log-Euclidean maps are flat in matrix-log coordinates; tangents still use ambient SPD coefficients
    {
        const LogEuclideanGeometry<SPD> geometry;
        const auto tangent = geometry.logarithm(p, q);
        const SPD recovered = geometry.exponential(p, tangent);
        const double distance = geometry.distance(p, q);
        const double squared_norm = geometry.inner_product(p, tangent, tangent);
        // exponentiating the initial velocity reconstructs the supplied endpoint, including its off-diagonal entry
        fdapde_strong_assert(error(recovered, q) < 1e-11, std::runtime_error, "LE exp-log endpoint mismatch");
        // a minimizing geodesic's initial squared norm equals its endpoint distance squared
        fdapde_strong_assert(
          std::abs(squared_norm - distance * distance) < 1e-11, std::runtime_error, "LE metric mismatch");
        const auto transported = geometry.transport(p, q, tangent);
        // parallel transport preserves the same tangent length when measured at the target
        fdapde_strong_assert(
          std::abs(geometry.norm(q, transported) - distance) < 1e-11, std::runtime_error, "LE transport mismatch");
        const auto mean = weighted_karcher_mean(geometry, samples, weights);
        // the exact log mean reports success before its point is used
        fdapde_strong_assert(mean.converged(), std::runtime_error, "LE mean did not converge");
        // commuting diagonal logarithms reduce to independent scalar geometric means
        fdapde_strong_assert(error(mean.point, geometric_mean) < 1e-11, std::runtime_error, "LE mean oracle mismatch");
        report("LE", distance, squared_norm, mean);

        // weighted_mean is a deferred algebraic log combination and does not normalize its weights
        const Vector<double, 2> coefficients {1., 3.};
        const SPD unnormalized(geometry.weighted_mean(samples, coefficients));
        const SPD algebraic_oracle(Vector<double, 3> {64., 0., 2916.});
        // exp(log(A)+3*log(B)) is diag(1*4^3,4*9^3), distinct from a normalized barycenter
        fdapde_strong_assert(
          error(unnormalized, algebraic_oracle) < 1e-9, std::runtime_error, "LE algebraic weight mismatch");
        std::cout << "LE weighted_mean with raw (1,3): diag(" << unnormalized(0, 0) << ", " << unnormalized(1, 1)
                  << "); this operation does not normalize\n";
    }

    // affine-invariant maps whiten the base point; its optimizer retraction differs from this exact exponential
    {
        const AffineInvariantGeometry<SPD> geometry;
        const auto tangent = geometry.logarithm(p, q);
        const SPD recovered = geometry.exponential(p, tangent);
        const double distance = geometry.distance(p, q);
        const double squared_norm = geometry.inner_product(p, tangent, tangent);
        // the relative spectral exponential reconstructs the original noncommuting endpoint
        fdapde_strong_assert(error(recovered, q) < 1e-11, std::runtime_error, "AIRM exp-log endpoint mismatch");
        // whitening gives the same squared length as the pair distance
        fdapde_strong_assert(
          std::abs(squared_norm - distance * distance) < 1e-11, std::runtime_error, "AIRM metric mismatch");
        const auto transported = geometry.transport(p, q, tangent);
        // congruence transport preserves length under the metric at q
        fdapde_strong_assert(
          std::abs(geometry.norm(q, transported) - distance) < 1e-11, std::runtime_error, "AIRM transport mismatch");
        const auto mean = weighted_karcher_mean(geometry, samples, weights, initial, options);
        // an iterative mean candidate is usable only after its stationarity check succeeds
        fdapde_strong_assert(mean.converged(), std::runtime_error, "AIRM mean did not converge");
        // diagonal AIRM means agree with scalar geometric means even from an unrelated identity initializer
        fdapde_strong_assert(error(mean.point, geometric_mean) < 1e-8, std::runtime_error, "AIRM mean oracle mismatch");
        report("AIRM", distance, squared_norm, mean);
    }

    // the Bures-Wasserstein metric interpolates diagonal square roots; large extrapolations have a local domain
    {
        const BuresWassersteinGeometry<SPD> geometry;
        const auto tangent = geometry.logarithm(p, q);
        const SPD recovered = geometry.exponential(p, tangent);
        const double distance = geometry.distance(p, q);
        const double squared_norm = geometry.inner_product(p, tangent, tangent);
        // the admissible endpoint step follows the horizontal lift back to q
        fdapde_strong_assert(error(recovered, q) < 1e-11, std::runtime_error, "BW exp-log endpoint mismatch");
        // the Sylvester metric gives the same squared length as the pair formula
        fdapde_strong_assert(
          std::abs(squared_norm - distance * distance) < 1e-11, std::runtime_error, "BW metric mismatch");
        const auto mean = weighted_karcher_mean(geometry, samples, weights, options);
        // the default arithmetic initializer is followed by certified iterative descent
        fdapde_strong_assert(mean.converged(), std::runtime_error, "BW mean did not converge");
        // averaging scalar square roots and squaring differs from the geometric-mean oracle
        fdapde_strong_assert(error(mean.point, root_mean) < 1e-8, std::runtime_error, "BW mean oracle mismatch");
        report("BW", distance, squared_norm, mean);
    }

    // log-Cholesky geometry is flat in lower-factor coordinates with logarithmic diagonal entries
    {
        const LogCholeskyGeometry<SPD> geometry;
        const auto tangent = geometry.logarithm(p, q);
        const SPD recovered = geometry.exponential(p, tangent);
        const double distance = geometry.distance(p, q);
        const double squared_norm = geometry.inner_product(p, tangent, tangent);
        // the inverse triangular chart reconstructs q from its initial ambient velocity
        fdapde_strong_assert(error(recovered, q) < 1e-11, std::runtime_error, "LC exp-log endpoint mismatch");
        // the flat chart metric matches the endpoint distance
        fdapde_strong_assert(
          std::abs(squared_norm - distance * distance) < 1e-11, std::runtime_error, "LC metric mismatch");
        const auto transported = geometry.transport(p, q, tangent);
        // the chart differential is held constant by the parallel transport
        fdapde_strong_assert(
          std::abs(geometry.norm(q, transported) - distance) < 1e-11, std::runtime_error, "LC transport mismatch");
        const auto mean = weighted_karcher_mean(geometry, samples, weights);
        // the exact flat-chart mean reports its closed-form convergence certificate
        fdapde_strong_assert(mean.converged(), std::runtime_error, "LC mean did not converge");
        // diagonal Cholesky factors recover the scalar geometric means after squaring
        fdapde_strong_assert(error(mean.point, geometric_mean) < 1e-11, std::runtime_error, "LC mean oracle mismatch");
        report("log-Cholesky", distance, squared_norm, mean);
    }

    // the Cheeger metric adds a rotation penalty rho=epsilon^2 to the logarithmic chart alignment
    {
        const CheegerLogEuclideanGeometry<SPD> geometry(1.);
        const auto pair = geometry.pair(p, q);
        // this example uses a resolved planar pair before requesting its single-valued logarithm
        fdapde_strong_assert(pair.unique(), std::runtime_error, "C-LE pair has competing minimizing rotations");
        const auto tangent = geometry.logarithm(p, q);
        const SPD recovered = geometry.exponential(p, tangent);
        const double distance = geometry.distance(p, q);
        const double squared_norm = geometry.inner_product(p, tangent, tangent);
        // the selected quotient-geodesic velocity reconstructs the original endpoint
        fdapde_strong_assert(error(recovered, q) < 1e-9, std::runtime_error, "C-LE exp-log endpoint mismatch");
        // the quotient metric agrees with the selected pair's squared length
        fdapde_strong_assert(
          std::abs(squared_norm - distance * distance) < 1e-9, std::runtime_error, "C-LE metric mismatch");
        const auto mean = weighted_karcher_mean(geometry, samples, weights, initial, options);
        // the generic solver requires an explicit initial point and a convergence check
        fdapde_strong_assert(mean.converged(), std::runtime_error, "C-LE mean did not converge");
        // these equally ordered diagonal charts need no rotation and have the scalar geometric-mean solution
        fdapde_strong_assert(error(mean.point, geometric_mean) < 1e-8, std::runtime_error, "C-LE mean oracle mismatch");
        // numerical stationarity does not provide a general global uniqueness certificate
        fdapde_strong_assert(
          mean.uniqueness == BarycenterUniqueness::not_certified, std::runtime_error, "unexpected C-LE certificate");
        report("C-LE rho=1", distance, squared_norm, mean);
        std::cout << "C-LE mean: stationary candidate; global uniqueness is not certified by this solver\n";
    }
    std::cout << "Transport shown for LE, AIRM and log-Cholesky; BW and C-LE do not expose that operation.\n"
              << "All endpoint, metric, transport and diagonal-mean checks passed.\n";
}
