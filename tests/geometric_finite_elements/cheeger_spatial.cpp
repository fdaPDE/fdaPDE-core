// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/finite_elements.h>
#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using Geometry = manifold::CheegerLogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
using Point = Geometry::Point;
using Batch = MatrixBatch<Point>;
/// @brief compares complete symmetric matrices in ambient coordinates
double error(const auto& a, const auto& b) { return Matrix<double, 2, 2>(a - b).norm(); }
/// @brief supplies two triangles with different local vertex orders
Triangulation<2, 2> mesh() {
    Eigen::Matrix<double, 4, 2> vertices;
    vertices << 0, 0, 1, 0, 0, 1, 1, 1;
    Eigen::Matrix<int, 2, 3> cells;
    cells << 2, 0, 1, 1, 3, 2;
    return {vertices, cells, Eigen::Matrix<int, 4, 1>::Ones()};
}
/// @brief supplies nearby noncommuting tensors with a stable local minimum
Batch values() {
    Batch data(4);
    data[0] = Geometry::from_chart({.1, .3, .1});
    data[1] = Geometry::from_chart({.2, .25, .2});
    data[2] = Geometry::from_chart({-.1, .4, -.1});
    data[3] = Geometry::from_chart({.3, .2, .3});
    return data;
}
/// @brief resolves stationarity well below the spatial finite-difference accuracy
gfe::P1GeodesicLinearizationOptions accurate() {
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.gradient_tolerance = 1e-11;
    return options;
}
/// @brief checks scalar DOF ownership, variable-rho spatial derivatives and prepared parallel evaluation
TEST(CheegerSpatial, OwnedP1RhoAndReusableEvaluation) {
    const auto domain = mesh();
    auto rho = std::array {.2, .4, .3, .6};
    const GeometricFeSpace space(domain, P1<1>, Geometry {}, std::span<const double>(rho));
    const auto original = rho;
    rho[0] = 100;
    // the space snapshots rho, so external mutation cannot invalidate prepared geometry
    EXPECT_EQ(space.rho_coefficients()[0], original[0]);
    const auto data = values();
    GeometricFeFunction function(space, data, accurate());
    const Eigen::Vector2d x(.2, .3);
    const auto fit = function.linearization(x);
    // the first cell uses local order (2,0,1), giving the continuous scalar value at x
    EXPECT_NEAR(fit.geometry().rho(), .3 * .3 + .5 * .2 + .2 * .4, 1e-14);
    const auto expected = gfe::p1_geodesic_value(
      Geometry::from_rho(.27), data.select(std::array {2, 0, 1}), std::array {.3, .5, .2}, accurate().mean);
    // field evaluation shares matrix DOFs and basis weights with the scalar rho coefficient field
    EXPECT_LT(error(Point(function(x)), expected.value), 1e-10);
    constexpr double h = 1e-5;
    const Point plus(function(Eigen::Vector2d(x[0] + h, x[1]))), minus(function(Eigen::Vector2d(x[0] - h, x[1])));
    const Geometry::Tangent difference((plus - minus) / (2 * h));
    // the total x derivative includes drho/dx=.2 through local weight direction (0,-1,1)
    EXPECT_LT(error(fit.weight_jvp(std::array {0., -1., 1.}).derivative, difference), 1e-6);
    MatrixBatch<Vector<double, 2>> locations(3);
    locations[0] = Vector<double, 2> {.2, .3};
    locations[1] = Vector<double, 2> {.8, .7};
    locations[2] = Vector<double, 2> {.5, .5};
    const auto plan = space.prepare_evaluation(locations);
    const auto seq = plan(function, execution_seq), par = plan(function, execution_par);
    for (std::size_t i = 0; i < seq.size(); ++i) {
        // parallel preparation and evaluation preserve point order and fixed-rho numerical results
        EXPECT_LT(error(seq[i], par[i]), 1e-13);
        const Eigen::Vector2d point(locations[i][0], locations[i][1]);
        // prepared spatial metadata produces the same value as an independent location lookup
        EXPECT_LT(error(seq[i], Point(function(point))), 1e-11);
    }
    const auto left =
      Geometry::from_rho(.35).interpolant(domain.cell(0), data.select(std::array {2, 0, 1}), accurate());
    const auto right =
      Geometry::from_rho(.35).interpolant(domain.cell(1), data.select(std::array {1, 3, 2}), accurate());
    const Eigen::Vector2d shared(.5, .5);
    // reordered cells agree on a shared edge because they share its nodal rho interpolation
    EXPECT_LT(error(Point(left(shared)), Point(right(shared))), 1e-11);
    // the variable-rho field uses that same edge value rather than the geometry's default .25
    EXPECT_LT(error(Point(function(shared)), Point(left(shared))), 1e-11);
    auto replacement = data;
    for (std::size_t i = 0; i < replacement.size(); ++i) replacement[i] = Geometry::from_chart({.4, .2, .1});
    function.set_coeff(replacement);
    const auto updated = plan(function);
    // the reusable plan reads new matrices after coefficient-dependent caches are invalidated
    EXPECT_LT(error(updated[0], replacement[0]), 1e-12);
}
/// @brief checks constant-field equivalence and public scalar-field validation
TEST(CheegerSpatial, ConstantAndInvalidRho) {
    const auto domain = mesh();
    const GeometricFeSpace constant(domain, P1<1>, Geometry {});
    const GeometricFeSpace nodal(domain, P1<1>, Geometry {}, std::span<const double>(std::array {.25, .25, .25, .25}));
    GeometricFeFunction f(constant, values(), accurate()), g(nodal, values(), accurate());
    // a constant geometry has no per-node scalar allocation
    EXPECT_TRUE(constant.rho_coefficients().empty());
    const Eigen::Vector2d x(.2, .3);
    // a constant P1 coefficient field reduces to the fixed-rho implementation
    EXPECT_LT(error(Point(f(x)), Point(g(x))), 1e-12);
    const auto invalid = [&](std::span<const double> rho) {
        const GeometricFeSpace space(domain, P1<1>, Geometry {}, rho);
    };
    // nodal scalar data must cover exactly the scalar space DOFs
    EXPECT_THROW(invalid(std::array {.25, .25}), std::invalid_argument);
    // nodal positivity guarantees positive rho throughout every convex P1 cell
    EXPECT_THROW(invalid(std::array {.25, 0., .25, .25}), std::invalid_argument);
}
/// @brief retains the TSPDE nodal state that previously produced false ties from unfinished starts
TEST(CheegerSpatial, CapturedUnfinishedCandidateRegression) {
    const auto domain = Triangulation<2, 2>::UnitSquare(3);
    const GeometricFeSpace space(domain, P1<1>, Geometry {});
    const std::array<double, 27> logs {
      1.8455488404708391,  0.12715912181951602, -0.9092253927076908, 0.8964963979510548,   -0.16971611821107793,
      -1.2127779331606146, -0.2716594345598309, 0.3669867113339678,  -0.8961476981679555,  -0.3441704318300115,
      0.1849420471565558,  0.6737500284373914,  0.7235572396297614,  0.062425435292336906, -0.08197923457306543,
      -0.799716278798453,  -0.5432160657333651, 0.3263528740060933,  -1.0573555777876873,  0.08255232496837742,
      -0.9269067783785264, 1.3958088224684027,  -0.4277605072793869, -0.5961061063655503,  1.4935356385779919,
      0.2149849639532032,  -0.6824846766799997};
    Batch data(9);
    for (int i = 0; i < 9; ++i) {
        SymmetricMatrix<double, 2> log;
        log(0, 0) = logs[3 * i];
        log(0, 1) = logs[3 * i + 1];
        log(1, 1) = logs[3 * i + 2];
        data[i] = matrix_exp(log);
    }
    auto options = accurate();
    options.mean.solver.gradient_tolerance = 1e-10;
    options.mean.solver.max_iterations = 500;
    const GeometricFeFunction function(space, std::move(data), options);
    const std::array<std::array<double, 2>, 40> locations {
      {{0.486667179735377, 0.541803933912888},  {0.191365255275741, 0.00907950708642602},
       {0.993271879851818, 0.390592301730067},  {0.14670268422924, 0.854989633196965},
       {0.241589481011033, 0.100678574759513},  {0.537101219408214, 0.504788867896423},
       {0.358212352963164, 0.825599786359817},  {0.871918980265036, 0.294746535364538},
       {0.392591055715457, 0.804755301447585},  {0.216567251831293, 0.2156979276333},
       {0.793461994733661, 0.789890799438581},  {0.260072831297293, 0.652055075392127},
       {0.268315598135814, 0.0651057146023959}, {0.535648625111207, 0.379914573626593},
       {0.291421601781622, 0.418455113191158},  {0.948105040937662, 0.187519100029022},
       {0.0635287179611623, 0.201817091554403}, {0.0913396079558879, 0.890405579702929},
       {0.310976799810305, 0.332529607694596},  {0.768619867041707, 0.894432441564277},
       {0.397760117659345, 0.264172520488501},  {0.969934918917716, 0.86933708935976},
       {0.380703847389668, 0.141835248330608},  {0.612551142228767, 0.00914845569059253},
       {0.24757822509855, 0.433234924683347},   {0.277621288085356, 0.305674390168861},
       {0.344674278749153, 0.33644685219042},   {0.411044177366421, 0.101568668615073},
       {0.57036917284131, 0.645194985205308},   {0.0170132112689316, 0.618044289760292},
       {0.0845533474348485, 0.593218012945727}, {0.708201471716166, 0.59875531680882},
       {0.179861813783646, 0.599496515700594},  {0.139522276818752, 0.387194500537589},
       {0.72047842037864, 0.796022088034078},   {0.762117812875658, 0.627067471388727},
       {0.370232261484489, 0.817295492626727},  {0.395980028202757, 0.168652257882059},
       {0.312913163565099, 0.523665966000408},  {0.431348290527239, 0.452453426085413}}
    };
    for (const auto& location : locations) {
        const auto fit = function.linearization(Eigen::Vector2d(location[0], location[1]));
        // only stationary competing candidates may mark these resolved captured means ambiguous
        ASSERT_TRUE(fit.result().converged() && !fit.result().detected_ambiguity);
        // the corrected branch selection must also permit the smoothing derivative on each training site
        EXPECT_TRUE(fit.rho_jvp().converged());
    }
}
}   // namespace
