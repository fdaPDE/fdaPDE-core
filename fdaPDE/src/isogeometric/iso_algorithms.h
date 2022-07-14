#ifndef __ISO_ALGORITHMS_H__
#define __ISO_ALGORITHMS_H__
#include "header_check.h"

namespace fdapde {
    // Simple data structure to hold the NURBS mesh data
    // This structure is used to pass the NURBS mesh data to the algorithms
    template <int LocalDim> 
    struct IsoMeshData {
        std::array<std::vector<double>, LocalDim> knots;
        MdArray<double, full_dynamic_extent_t<LocalDim>> weights;
        MdArray<double, full_dynamic_extent_t<LocalDim + 1>> control_points;
        std::array<int, LocalDim> degree;
        int flags = 0;
    
        IsoMeshData(std::array<std::vector<double>, LocalDim> knots,
                    MdArray<double, full_dynamic_extent_t<LocalDim>> weights,
                    MdArray<double, full_dynamic_extent_t<LocalDim + 1>> control_points,
                    std::array<int, LocalDim> degree,
                    int flags = 0)
            : knots(std::move(knots)),
            weights(std::move(weights)),
            control_points(std::move(control_points)),
            degree(std::move(degree)),
            flags(flags) {}

        // constructor for 1D
        IsoMeshData(std::vector<double> knots,
                    MdArray<double, full_dynamic_extent_t<1>> weights,
                    MdArray<double, full_dynamic_extent_t<2>> control_points,
                    int degree,
                    int flags = 0)
            : knots({knots}),
            weights(std::move(weights)),
            control_points(std::move(control_points)),
            degree({degree}),
            flags(flags) {}
        };


}

namespace fdapde {
namespace iso_algorithms {

constexpr double pi = 3.14159265358979323846;
constexpr double tol = 1e-9;

typedef Eigen::Matrix<double, 3, 1> Point3D;
typedef Eigen::Matrix<double, 2, 1> Point2D;


// remember to add sanity checks

Point3D point_to_ray(const Point3D& origin, const Point3D& direction, Point3D& point) {
    return origin + direction * (point - origin).dot(direction);
}

// non c'é
Point3D point_to_line(const Point3D& start, const Point3D& end, Point3D& point) {
    double param = (point - start).dot(end - start) /  (end - start).dot(end - start);
    return start + param * (end - start);
}

Point3D compute_rays(const Point3D& point0, const Point3D& vector0, const Point3D& point1, const Point3D& vector1) {
    // check that the vectors are not zero
    fdapde_assert(vector0.norm() > 0 && vector1.norm() > 0);

    auto cross = vector0.cross(vector1);
    auto diff = point1 - point0;
    auto coin_cross = diff.cross(vector1);

    if(cross.norm() == 0 && coin_cross.norm() == 0) {
        // the lines are coplanar
        std::cout << "Warning: The lines are coincident" << std::endl;
        return point0;
    } else if(cross.norm() == 0 && coin_cross.norm() != 0) {
        // the lines are parallel
        std::cout << "Warning: The lines are parallel" << std::endl;
        return point0;
    } else {
        // the lines intersect or are skew
        double cross_norm = cross.norm();
        
        auto pd1_cross = diff.cross(vector1);
        double param0 = pd1_cross.dot(cross) / (cross_norm * cross_norm);

        auto pd2_cross = diff.cross(vector0);
        double param1 = pd2_cross.dot(cross) / (cross_norm * cross_norm);

        Point3D rayP0 = point0 + param0 * vector0;
        Point3D rayP1 = point1 + param1 * vector1;

        if ((rayP0 - rayP1).norm() < tol) {
            return rayP0;
        } else {
            std::cout << "Warning: The lines are skew" << std::endl;
            return (rayP0 + rayP1) / 2.0;
        }
    }
}

// Implementation of algorihm 8.1 of the NURBS book
// We need a curve in 3D to create a surface
IsoMeshData<2> create_revolved_ISO_surface(const IsoMeshData<1>& curve, double rad, 
    const Point3D& axis = Point3D(0,1,0), const Point3D& origin = Point3D::Zero()) {
    // crea una superficie ISO ruotata a partire da una curva ISO
    // curve: curva ISO
    // angle: angolo di rotazione in radianti
    // origin: punto di origine della rotazione
    // axis: asse di rotazione

    fdapde_assert(axis.norm() > 0);

    std::vector<double> knotVectorU;

    int narcs = 0;

    if(rad <= pi/2 ){
        narcs = 1;
        knotVectorU.resize(2 * narcs + 3 + 1);
    } else if(rad <= pi) {
        narcs = 2;
        knotVectorU.resize(2 * narcs + 3 + 1);
		knotVectorU[3] = knotVectorU[4] = 0.5;
    } else if(rad <= 3*pi/2) {
        narcs = 3;
        knotVectorU.resize(2 * narcs + 3 + 1);
        knotVectorU[3] = knotVectorU[4] = 1.0 / 3.0;
        knotVectorU[5] = knotVectorU[6] = 2.0 / 3.0;
    } else {
        narcs = 4;
        knotVectorU.resize(2 * narcs + 3 + 1);
        knotVectorU[3] = knotVectorU[4] = 0.25;
        knotVectorU[5] = knotVectorU[6] = 0.5;
        knotVectorU[7] = knotVectorU[8] = 0.75;
    }

    double dtheta = rad/narcs;
    int jj = 3 + 2*(narcs - 1);

    for (int i = 0; i < 3; i++)
	{
		knotVectorU[i] = 0.0;
		knotVectorU[jj + i] = 1.0;
	}

	int n = 2 * narcs;
	double wm = std::cos(dtheta / 2.0);
	double angle = 0.0;
	std::vector<double> cosines(narcs + 1, 0.0);
	std::vector<double> sines(narcs + 1, 0.0);

    for (int i = 1; i <= narcs; i++)
	{
		angle += dtheta;
		cosines[i] = std::cos(angle);
		sines[i] = std::sin(angle);
	}

    int m = curve.weights.extent(0) - 1;
    Point3D X, Z, O, P0, P2, T0, T2;
    double r = 0.0;
    int index = 0;

    int degreeU = 2;
    MdArray<double, MdExtents<Dynamic,Dynamic>> weights(n + 1, m + 1);
    MdArray<double, MdExtents<Dynamic, Dynamic,Dynamic>> control_points(n + 1, m + 1, 3);

    //std::cout << "m: " << m << std::endl;


    for (int i = 0; i <= m; i++){
        Point3D P(3);
        for (int j = 0; j < 3; j++){
            P(j) = curve.control_points(i, j);
        }
        Point3D O = point_to_ray(origin, axis, P);
        Point3D X = P - O;

        r = X.norm();
        X = X.normalized();
        Point3D Y = axis.cross(X)/r; 

        P0 = P;

        for(int h=0;h<3;h++){
            control_points(0, i, h) = P0(h);
            weights(0, i) = curve.weights(i);
        }


        T0 = Y;
        index = 0;

        for(int m=1; m<=narcs; m++){
            P2 = (r < 1e-8) ? O : O + r * (cosines[m] * X + sines[m] * r * Y);
            
            for(int h=0; h < 3; h++) control_points(index + 2, i, h) = P2(h);
            weights(index + 2, i) = curve.weights(i);

            T2 = cosines[m] * Y - sines[m] * X;


            if (r < tol){
                for(int h=0; h<3; h++) control_points(index + 1, i, h) = O(h);
                weights(index + 1, i) = wm * curve.weights(i);
            } else {
                // Find intersect point
                Point3D intersect_point = compute_rays(P0, T0, P2, T2);

                fdapde_assert((intersect_point - P0).norm() > tol);

                for(int h=0; h<3; h++) control_points(index + 1, i, h) = intersect_point(h);
                weights(index + 1, i) = wm * curve.weights(i);
            }

            index += 2;
            if(m < narcs){
                P0 = P2;
                T0 = T2;
            }
        }
        

    }
    //std::cout << "n: " << n << std::endl;

    std::array<std::vector<double>, 2> new_knots;

    new_knots[0] = knotVectorU;
    new_knots[1] = curve.knots[0];

    std::array<int, 2> new_order = {degreeU, curve.degree[0]};
    //std::cout << "new_order: " << new_order[0] << " " << new_order[1] << std::endl;

    return IsoMeshData<2>(new_knots, weights, control_points, new_order);
}



IsoMeshData<1> knots_refinement(const IsoMeshData<1>& mesh_data, std::vector<double> refinement_knots){
// ALGO A5.4 pag. 164 NURBS book for 1D knot refinement

    auto old_cp = mesh_data.control_points;
    auto old_w = mesh_data.weights;
    auto knots_ = mesh_data.knots[0];
    auto order_ = mesh_data.degree[0];
    int EmbedDim = old_cp.extent(1);

    MdArray<double, MdExtents<Dynamic,Dynamic>> new_cp;
    MdArray<double, MdExtents<Dynamic>> new_w;
    std::vector<double> updated_knots;

    new_cp.resize(old_cp.extent(0) + refinement_knots.size(),EmbedDim);
    new_w.resize(old_cp.extent(0) + refinement_knots.size());
    updated_knots.resize(knots_.size() + refinement_knots.size());
                
    // get the number of control points
    int n = old_cp.extent(0) - 1;
    int m = order_ + n + 1;

    int r = refinement_knots.size() - 1;

    // get the span
    auto old_basis  = BSplineBasis(knots_, order_);
    int a = old_basis.find_span(refinement_knots[0], n);
    int b = old_basis.find_span(refinement_knots[r], n) + 1 ;


    // get the new control points
    for(int j=0; j<=a-order_; j++) {
        new_w(j) = old_w(j);
        for(int i=0; i<EmbedDim; i++) new_cp(j,i) = old_cp(j,i);
    }

    for(int j=b-1; j<=n; j++) {
        new_w(j+r+1) = old_w(j);
        for(int i=0; i<EmbedDim; i++) new_cp(j+r+1,i) = old_cp(j,i);
    }

   // get the new knots
    for(int j=0; j<=a; j++) updated_knots[j] = knots_[j];
    for(int j=b+order_; j<=m; j++) updated_knots[j+r+1] = knots_[j]; 

    // get the new control points
    int ii = b + order_ - 1;
    int kk = b + order_ + r;


    for(int j=r; j>=0; j--) {
        while(refinement_knots[j] <= knots_[ii] && ii > a) {
            new_w(kk-order_-1) = old_w(ii-order_-1);
            for(int h=0; h<EmbedDim; h++) new_cp(kk-order_-1,h) = old_cp(ii-order_-1,h);
            updated_knots[kk] = knots_[ii];
            kk = kk - 1;
            ii = ii - 1;
        }
        
        new_w(kk-order_-1) = new_w(kk-order_);
        for(int h =0; h < EmbedDim; h++) new_cp(kk-order_-1,h) = new_cp(kk-order_,h);

        for(int l = 1; l<=order_; l++) {
            int ind = kk-order_+l;
            double alpha = updated_knots[kk+l] - refinement_knots[j];
            if(alpha == 0.0) {
                new_w(ind-1) = new_w(ind);
                for(int h=0; h<EmbedDim; h++) new_cp(ind-1,h) = new_cp(ind,h);
            } else {
                alpha = alpha / (updated_knots[kk+l] - knots_[ii-order_+l]);         
                for(int h=0; h<EmbedDim; h++) new_cp(ind-1,h) = (alpha * new_w(ind-1)*new_cp(ind-1,h) + 
                                                (1.0 - alpha) * new_w(ind)*new_cp(ind,h))/(alpha*new_w(ind-1) + (1.0 - alpha )*new_w(ind));
                new_w(ind-1) = alpha * new_w(ind-1) + (1.0 - alpha) * new_w(ind);
            }
        }
        updated_knots[kk] = refinement_knots[j];
        kk = kk - 1;  
    }

    std::array<std::vector<double>, 1> new_knots = {updated_knots};
    std::array<int, 1> new_order = {order_};

    ////// end of ALGO A5.5
    return IsoMeshData<1>(new_knots, new_w, new_cp, new_order);
}

} // namespace iso_algorithms
} // namespace fdapde

#endif // __ISO_ALGORITHMS_H__