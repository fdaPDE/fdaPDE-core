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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include <fdaPDE/geometric_discretization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using Dense = Matrix<double, 3, 3>;
using Tangent = SymmetricMatrix<double, 3>;
using Geometry = manifold::BuresWassersteinSPDGeometry<double, 3>;
using Point = Geometry::Point;
using Coefficients = std::array<double, 6>;

/// @brief stores exact binary64 input and a solution generated independently with 80-digit Decimal arithmetic
struct Fixture {
    const char* name;
    Coefficients mean, rhs, golden_solution;
    std::array<Coefficients, 3> nodes;
    std::array<double, 3> weights;
};

// golden solutions use 80-digit Jacobi square roots and full Sylvester Gaussian elimination
// binary64 fixture inputs are converted exactly before evaluating the mean Hessian
const std::array<Fixture, 5> fixtures {
  {{"original_adjoint",
    {{3.9616912471183, 1.0136426808023602, -0.0968287435582722, 0.7388076358882731, 0.2233915295836809,
      0.13104714211799812}},
    {{-0.05458146997323126, -0.004940820318118876, 0.015967488872621043, -0.00046766537788926326, 0.004223182469766965,
      2.2093700520890427e-05}},
    {{-0.05417789360162242, -0.0051109080637083875, 0.016501504317445837, -0.0008413868362512314, 0.004315054845311631,
      -7.761212725979936e-06}},
    {{{{2.259883884038278, -1.6113426298206204, 0.7006750290011167, 2.3379657203194752, -1.766791294479571,
        1.5677262433495789}},
      {{4.018916185066093, -0.33187881914556044, 0.2690362020774514, 0.4305967815985394, -0.16178477256396676,
        0.0663243924290517}},
      {{4.246907865861416, 1.668037788498956, -0.10797866464291693, 0.9888283655330448, 0.3535571313600276,
        0.4726370534885115}}}},
    {{0.08332982938736389, 0.17755698738620027, 0.7391131832264358}}},
   {"second_adjoint",
    {{3.9616912471183, 1.0136426808023602, -0.0968287435582722, 0.7388076358882731, 0.2233915295836809,
      0.13104714211799812}},
    {{0.21272909167238807, 0.22141347168687806, 0.21096944164097056, 0.14215855474195666, 0.08537848993804097,
      0.01485072804888293}},
    {{0.2215140205592462, 0.21784195691470049, 0.22276912670136098, 0.13343003835649445, 0.08728904457987352,
      0.014794315547487011}},
    {{{{2.259883884038278, -1.6113426298206204, 0.7006750290011167, 2.3379657203194752, -1.766791294479571,
        1.5677262433495789}},
      {{4.018916185066093, -0.33187881914556044, 0.2690362020774514, 0.4305967815985394, -0.16178477256396676,
        0.0663243924290517}},
      {{4.246907865861416, 1.668037788498956, -0.10797866464291693, 0.9888283655330448, 0.3535571313600276,
        0.4726370534885115}}}},
    {{0.08332982938736389, 0.17755698738620027, 0.7391131832264358}}},
   {"fold_1",
    {{3.8943827560962263, 1.067770836578567, -0.06438676422446947, 0.8889100515278208, 0.20595312111543662,
      0.08580793877030474}},
    {{-0.07015395441312625, -0.028632967603502316, 0.00234334296027119, -0.012647376181188764, -8.69132563017097e-07,
      -0.00034244300211403215}},
    {{-0.07048916718941703, -0.029691027412170154, 0.00336899414640316, -0.012111992968413662, -1.1601176315640657e-05,
      -0.0005426134385983462}},
    {{{{2.476357051065509, -1.229178498249558, 0.7961621667274982, 0.6112657405736797, -0.4531317929081905,
        3.1914312458171277}},
      {{4.044460618885156, -0.35131940236003084, 0.07946973995259884, 0.030521981944979137, -0.008253944750141264,
        0.43119592024751974}},
      {{4.044919954845153, 1.7680053428139983, -0.2741572143979421, 1.8184900694913602, -0.0643890397215402,
        0.021521650058585005}}}},
    {{0.08332982938736389, 0.17755698738620027, 0.7391131832264358}}},
   {"fold_2",
    {{3.9621718706618334, 1.0129983763086075, -0.09620306518381858, 0.7366498668759286, 0.22645769696634835,
      0.13453882736062145}},
    {{0.0008531240766707207, -0.000980750914299691, -0.000668016074594777, -0.000720424106070686,
      -0.00041019594410058425, -0.0002072260897427342}},
    {{0.0008710828558485871, -0.0009965614129081584, -0.0006494961645280552, -0.0007258181067609137,
      -0.0004048306882767724, -0.00021979086823037286}},
    {{{{2.2607446423440285, -1.6111665923369571, 0.6929901121520514, 2.3521998339386747, -1.759914268450825,
        1.5437366130826533}},
      {{4.016525891983125, -0.3302347771972892, 0.2699061656086128, 0.43211041256299615, -0.1610642403651617,
        0.06576220110913013}},
      {{4.24789976710251, 1.6658565797386264, -0.10671259333141875, 0.9839624198536973, 0.3552527798359086,
        0.47955827386687777}}}},
    {{0.08332982938736389, 0.17755698738620027, 0.7391131832264358}}},
   {"fold_4",
    {{4.626556203992334, 1.4719973691083694, 0.1561602851154933, 0.9630821497741198, 0.22363940625150325,
      0.06881433063177032}},
    {{-0.010258790537119046, 0.020208794876184877, 0.028962359119970874, 0.016348311062468226, 0.011615529496251947,
      0.0027574240012548565}},
    {{-0.010085002781916522, 0.020047032142820168, 0.030913314340438352, 0.016070389898264783, 0.012375470189144977,
      0.002861557410255776}},
    {{{{8.413269445225595, 3.373351178741111, -0.12800264735480882, 2.017868440859076, 1.079186977298315,
        1.922957939648763}},
      {{3.1649230629371834, 0.8847441793198773, -0.08153285849382277, 0.9150048459850323, -0.0637465734647278,
        0.006352651728945774}},
      {{3.4175447921441906, 0.8800517174097866, 0.319608633627971, 0.6111544304049699, -0.08389798389011505,
        0.10174545987537054}}}},
    {{0.2960615013726059, 0.12480899970977954, 0.5791294989176146}}}}
};

/// @brief restore raw symmetric coefficients without off-diagonal covector multiplicity
Tangent unpack(const Coefficients& values) {
    Tangent result;
    for (int row = 0, index = 0; row < 3; ++row)
        for (int column = row; column < 3; ++column, ++index) result(row, column) = values[index];
    return result;
}

/// @brief preserve a dense symmetric result without borrowing temporary owners
Tangent symmetric(const Dense& value) {
    Tangent result;
    for (int row = 0; row < 3; ++row)
        for (int column = row; column < 3; ++column)
            result(row, column) = .5 * (value(row, column) + value(column, row));
    return result;
}

/// @brief solve AX plus XA transpose through an independent full nine-coordinate system
Dense sylvester(const Dense& coefficient, const Dense& rhs) {
    Matrix<double, 9, 9> matrix;
    Vector<double, 9> vector;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column) {
            vector[3 * row + column] = rhs(row, column);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    matrix(3 * row + column, 3 * i + j) =
                      (j == column ? coefficient(row, i) : 0.) + (i == row ? coefficient(column, j) : 0.);
        }
    const Vector<double, 9> solution(PartialPivLU(matrix).solve(vector));
    Dense result;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column) result(row, column) = solution[3 * row + column];
    const Dense residual(coefficient * result + result * coefficient.transpose() - rhs);
    // the full-system oracle must solve its defining equation independently of the packed BW kernel
    EXPECT_LT(residual.norm() / std::max(1., rhs.norm()), 1e-11);
    return result;
}

/// @brief retain public distance frames for the three captured endpoints
std::vector<Geometry::RelativeFrame> frames(const Fixture& fixture, const Point& mean) {
    std::vector<Geometry::RelativeFrame> result;
    for (const auto& node : fixture.nodes) result.push_back(Geometry().relative_frame(mean, Point(unpack(node))));
    return result;
}

/// @brief apply the original weighted geometric mean Hessian without a matrix surrogate
Tangent native_action(
  const Fixture& fixture, const Point& mean, const std::vector<Geometry::RelativeFrame>& relative,
  const Tangent& direction) {
    Tangent result = Geometry().zero_tangent(mean);
    for (int i = 0; i < 3; ++i) {
        const Tangent value = Geometry().half_squared_distance_hessian_vector(relative[i], direction);
        result = Tangent(result + fixture.weights[i] * value);
    }
    return result;
}

/// @brief evaluate mean curvature from independent general Sylvester systems and its nonzero stationarity residual
Tangent independent_action(
  const Fixture& fixture, const Point& mean, const std::vector<Geometry::RelativeFrame>& relative,
  const Tangent& direction) {
    const Dense point(mean), input(direction);
    const IdentityMatrix<double, 3, 3> identity;
    Dense total, stationarity;
    total.set_zero();
    stationarity.set_zero();
    for (int i = 0; i < 3; ++i) {
        const Dense transport(relative[i].transport);
        const Dense coefficient(transport * point), rhs(transport * input * transport);
        const Dense solution = sylvester(coefficient, rhs);
        total = Dense(total + fixture.weights[i] * solution);
        stationarity = Dense(stationarity + fixture.weights[i] * (transport - identity));
    }
    const Dense lift = sylvester(point, input);
    return symmetric(Dense(
      point * total + total * point - input * stationarity - stationarity * input + lift * point * stationarity +
      stationarity * point * lift));
}

/// @brief checks isotropic identities without requiring unrepresentable optional Sylvester factors
template <int Order> void check_isotropic_range(int order) {
    using IsotropicGeometry = manifold::BuresWassersteinSPDGeometry<double, Order>;
    const IsotropicGeometry geometry = [&] {
        if constexpr (Order == Dynamic)
            return IsotropicGeometry(order);
        else
            return IsotropicGeometry();
    }();
    for (const double scale : {1., 1e280, 1e308}) {
        SCOPED_TRACE(scale);
        typename IsotropicGeometry::Tangent value;
        if constexpr (Order == Dynamic) value.resize(order, order);
        for (int row = 0; row < order; ++row)
            for (int col = row; col < order; ++col) value(row, col) = row == col ? scale : 0.;
        const typename IsotropicGeometry::Point point(value);
        const auto frame = geometry.relative_frame(point, point);
        // identical endpoints have zero distance even when a diagonal Sylvester coefficient exceeds binary64 range
        EXPECT_DOUBLE_EQ(geometry.distance(frame), 0.);
        const auto logarithm = geometry.logarithm(frame);
        // the self logarithm remains the exact zero tangent at every supported endpoint scale
        EXPECT_DOUBLE_EQ(logarithm.norm(), 0.);
        const auto action = geometry.logarithm_target_jvp(frame, value);
        for (int row = 0; row < order; ++row)
            for (int col = row; col < order; ++col) {
                // the target logarithm differential at coincident points is the identity, including spectral fallback
                EXPECT_DOUBLE_EQ(action(row, col) / scale, row == col ? 1. : 0.);
            }
        if constexpr (Order != Dynamic) {
            // only unrepresentable optional factors use the original spectral path
            EXPECT_EQ(frame.transport_factor.info(), scale > std::numeric_limits<double>::max() / 2 ? -1 : 0);
            // the Frobenius-adjoint factor obeys the same representability decision
            EXPECT_EQ(frame.transport_adjoint_factor.info(), scale > std::numeric_limits<double>::max() / 2 ? -1 : 0);
        }
    }
}

// optional small transport caches preserve the endpoint range for fixed and runtime SPD2/SPD3
TEST(BuresWassersteinConditioning, IsotropicExtremeScaleFallback) {
    check_isotropic_range<2>(2);
    check_isotropic_range<3>(3);
    check_isotropic_range<Dynamic>(2);
    check_isotropic_range<Dynamic>(3);
}

// conditioned BW transport derivatives agree with a full Sylvester oracle rather than amplified inverse congruences
TEST(BuresWassersteinConditioning, TransportAndGoldenMeanActions) {
    for (const auto& fixture : fixtures) {
        SCOPED_TRACE(fixture.name);
        const Point mean(unpack(fixture.mean));
        const auto relative = frames(fixture, mean);
        const Tangent rhs = unpack(fixture.rhs), golden = unpack(fixture.golden_solution);
        const double norm = Geometry().norm(mean, rhs);
        for (int i = 0; i < 3; ++i) {
            const Dense point(mean), transport(relative[i].transport), direction(golden);
            const Dense coefficient(transport * point), forcing(transport * direction * transport);
            const Dense solved = sylvester(coefficient, forcing);
            const Tangent oracle = symmetric(Dense(point * solved + solved * point));
            const Tangent actual = Geometry().logarithm_target_jvp(relative[i], symmetric(forcing));
            const Tangent error(actual - oracle);
            // the public target derivative must agree with the independently vectorized Sylvester equation
            EXPECT_LT(Geometry().norm(mean, error) / std::max(1., Geometry().norm(mean, oracle)), 1e-10);
        }
        const Tangent actual = native_action(fixture, mean, relative, golden);
        const Tangent oracle = independent_action(fixture, mean, relative, golden);
        const Tangent arithmetic_error(actual - oracle), golden_residual(actual - rhs);
        // native curvature must match the separately grouped full-system arithmetic at the original residual scale
        EXPECT_LT(Geometry().norm(mean, arithmetic_error) / norm, 1e-10);
        // the independent 80-digit solution certifies the true native operator, including finite mean residual terms
        EXPECT_LT(Geometry().norm(mean, golden_residual) / norm, 1e-10);
    }
}

// recursive CG convergence must also satisfy the unchanged true native residual on the captured right-hand sides
TEST(BuresWassersteinConditioning, NativeMeanSolvesPreserveOriginalTolerance) {
    for (const auto& fixture : fixtures) {
        SCOPED_TRACE(fixture.name);
        const Point mean(unpack(fixture.mean));
        const auto relative = frames(fixture, mean);
        const Tangent rhs = unpack(fixture.rhs), golden = unpack(fixture.golden_solution);
        manifold::PositiveDefiniteCGOptions options;
        options.max_iterations = 100;
        options.residual_tolerance = 1e-10;
        const auto result = manifold::PositiveDefiniteConjugateGradient(options).solve(
          [&](const Tangent& direction) { return native_action(fixture, mean, relative, direction); }, Geometry(), mean,
          rhs);
        // the original local CG must terminate under its inherited 100-iteration and 1e-10 limits
        ASSERT_TRUE(result.converged());
        const Tangent image = native_action(fixture, mean, relative, result.solution);
        const Tangent residual(image - rhs), solution_error(result.solution - golden);
        // recomputing Hx minus rhs catches residual gaps hidden by the recursive CG stopping certificate
        EXPECT_LE(Geometry().norm(mean, residual) / Geometry().norm(mean, rhs), 1e-10);
        // an independently generated solution checks the recovered tangent despite the conditioned mean operator
        EXPECT_LT(Geometry().norm(mean, solution_error) / Geometry().norm(mean, golden), 1e-8);
    }
}
}   // namespace
