#include <gtest/gtest.h> // testing framework
#include <limits>

#include "../../src/linear_algebra/vector_space.h"
#include "../../src/utils/symbols.h"
using fdapde::core::VectorSpace;
#include "utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;

TEST(VectorSpaceTest, projection1D) {
  // define a vector space by defining its direction
  SVector<2> i(1,1);
  VectorSpace<1, 2> vs({i}); // a line embedded in a 2D plane
  // its orthonormal basis is maden by the direction \tilde i = (1/sqrt(2), 1/sqrt(2))

  // take a 2D point and project it into vs
  SVector<2> p(4, 3.5);
  /* manual computations yeld to: \norm{\tilde i} = 1, (\tilde i \dot p) = 5,30330085889910643301;
     let P_i(p) denote the projection of p into the space spanned by \tilde i, we get
         P_i(p) = \frac{\tilde i \dot p}{\norm{\tilde i}} \tilde i = (3.75, 3.75)
     projection onto i instead is equal to
         Q_i(p) = \frac{\tilde i \dot p}{\norm{\tilde i}} = 5,30330085889910643301
   */
  EXPECT_TRUE((vs.project_into(p) - SVector<2>(3.75, 3.75)).norm() < DOUBLE_TOLERANCE);
  EXPECT_TRUE((vs.project_onto(p) - SVector<1>(5.30330085889910643301)).norm() < DOUBLE_TOLERANCE);
}

TEST(VectorSpaceTest, projection2D) {
  // define a vector space by defining its direction
  SVector<3> i(1,1,0), j(0,1,1);
  VectorSpace<2, 3> vs({i,j}); // a 2D plane embedded in a 3D space
  // orthonormal basis of vs is maden by the set {\tilde i, \tilde j} = {(1/sqrt(2), 1/sqrt(2), 0), (-1/sqrt(6), 1/sqrt(6), 2/sqrt(6))}

  // take a 3D point and project it into vs
  SVector<3> p(7.1, 3.4, 2);
  /* manual computations yeld to: \norm{\tilde i} = \norm{\tilde j} = 1,
     (\tilde i \dot p) = 7,42462120245874900621; (\tilde j \dot p) = 0,12247448713915890491;
     let P_i(p) denote the projection of p into the space spanned by {\tilde i, \tilde j}, we get
         P_i(p) = \frac{\tilde i \dot p}{\norm{\tilde i}} \tilde i + \frac{\tilde j \dot p}{\norm{\tilde j}} \tilde j =
	        = 7,42462120245874900621 * (1/sqrt(2), 1/sqrt(2), 0) + 0,12247448713915890491*(-1/sqrt(6), 1/sqrt(6), 2/sqrt(6)) =
		= (5.25, 5.25, 0) + (-0.05, 0.05, 0.1) = (5.20, 5.3, 0.1)
     projection onto i instead is equal to
         Q_i(p) = {\frac{\tilde i \dot p}{\norm{\tilde i}}, \frac{\tilde j \dot p}{\norm{\tilde j}}} =
	        = {7,42462120245874900621, 0,12247448713915890491}
   */
  EXPECT_TRUE((vs.project_into(p) - SVector<3>(5.20, 5.3, 0.1)).norm() < DOUBLE_TOLERANCE);
  EXPECT_TRUE((vs.project_onto(p) - SVector<2>(7.42462120245874900621,
					       0.12247448713915890491)).norm() < DOUBLE_TOLERANCE);
}

TEST(VectorSpaceTest, L2Distance1D) {
  // define a vector space by defining its direction
  SVector<2> i(1,2);
  VectorSpace<1, 2> vs({i}); // a line embedded in a 2D plane
  
  // the distance between a point in the space and the space itself should be zero
  EXPECT_NEAR(vs.distance(vs({1})), 0, DOUBLE_TOLERANCE);
  // the distance between a point projected into the space and the space itself should be zero
  SVector<2> p(7,9);
  EXPECT_NEAR(vs.distance(vs.project_into(p)), 0, DOUBLE_TOLERANCE);
  // expect .distance() to compute correct distance between q and its projection
  EXPECT_NEAR(vs.distance(p), (p - vs.project_into(p)).squaredNorm(), DOUBLE_TOLERANCE);
}

TEST(VectorSpaceTest, L2Distance2D) {
  // define a vector space by defining its direction
  SVector<3> i(1,2,10), j(7,7,5);
  VectorSpace<2, 3> vs({i,j}); // a place embedded in a 3D space
  
  // the distance between a point in the space and the space itself should be zero
  EXPECT_NEAR(vs.distance(vs({1})), 0, DOUBLE_TOLERANCE);
  // the distance between a point projected into the space and the space itself should be zero
  SVector<3> p(3,9,2);
  EXPECT_NEAR(vs.distance(vs.project_into(p)), 0, DOUBLE_TOLERANCE);
  // expect .distance() to compute correct distance between q and its projection
  EXPECT_NEAR(vs.distance(p), (p - vs.project_into(p)).squaredNorm(), DOUBLE_TOLERANCE);
}


TEST(VectorSpaceTest, AffineSpace) {
  // define a vector space by defining its direction
  SVector<3> i(1,1,0), j(0,1,1);
  SVector<3> offset(0,0,5);
  VectorSpace<2, 3> vs({i,j}, offset); // a plane embedded in a 3D space passing by (0,0,5)
  // orthonormal basis of vs is maden by the set {\tilde i, \tilde j} = {(1/sqrt(2), 1/sqrt(2), 0), (-1/sqrt(6), 1/sqrt(6), 2/sqrt(6))}

  // take a 3D point and project it into vs (we expect the same results of project2D corrected by (0,0,5))
  SVector<3> p(7.1, 3.4, 2);
  /* manual computations yeld to: \norm{\tilde i} = \norm{\tilde j} = 1,
     we need to compute the orthogonal projection of p - offset = (7.1, 3.4, -3)
     (\tilde i \dot p) = 7,42462120245874900621; (\tilde j \dot p) = -3,96000841749947125875;
     let P_i(p) denote the projection of p - offset into the space spanned by {\tilde i, \tilde j}, we get
         P_i(p) = \frac{\tilde i \dot p}{\norm{\tilde i}} \tilde i + \frac{\tilde j \dot p}{\norm{\tilde j}} \tilde j =
	        = 7,42462120245874900621 * (1/sqrt(2), 1/sqrt(2), 0) -3,96000841749947125875*(-1/sqrt(6), 1/sqrt(6), 2/sqrt(6)) =
		= (5.25, 5.25, 0) + (1,61666666666666666667, -1,61666666666666666667, -3,23333333333333333334)
		= (6,86666666666666666667, 3,63333333333333333333, -3,23333333333333333334)

     Then the orthogonal projection of p on the affine space is given by P_i(p) + offset
		
     projection onto i instead is equal to
         Q_i(p) = {\frac{\tilde i \dot p}{\norm{\tilde i}}, \frac{\tilde j \dot p}{\norm{\tilde j}}} =
	        = {7,42462120245874900621, -3,96000841749947125875}
   */
  EXPECT_TRUE((vs.project_into(p) - (SVector<3>(6.86666666666666666667,
						3.63333333333333333333,
						-3.23333333333333333334) + offset)).norm() < DOUBLE_TOLERANCE);
  EXPECT_TRUE((vs.project_onto(p) - SVector<2>(7.42462120245874900621,
					       -3.96000841749947125875)).norm() < DOUBLE_TOLERANCE);

  // the distance between a point in the space and the space itself should be zero
  EXPECT_NEAR(vs.distance(vs({1,2})), 0, DOUBLE_TOLERANCE);
  // the distance between a point projected into the space and the space itself should be zero
  SVector<3> q(3,9,2);
  EXPECT_NEAR(vs.distance(vs.project_into(q)), 0, DOUBLE_TOLERANCE);
  // expect .distance() to compute correct distance between q and its projection
  EXPECT_NEAR(vs.distance(q), (q - vs.project_into(q)).squaredNorm(), DOUBLE_TOLERANCE);
}
