#include <iostream>
#include <cmath>
#include <list>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib> 
#include <fstream>
#include <random> 
#include <nlohmann/json.hpp> 
using json = nlohmann::json;
#include "../geometry.h"
#include <chrono>
using namespace std::chrono;
using namespace fdapde;
using coords_t = Eigen::Matrix<double, 2, 1>;  // Vettore 2D

constexpr int N = 10; // Numero di punti del bordo esterno
constexpr double R = 1.0; // Raggio del dominio principale
constexpr double cx = 1.0, cy = 1.0; // Centro del dominio

constexpr int M = 8; // Numero di punti del buco
constexpr double r_hole = 0.4; // Raggio del buco
constexpr double cx_hole = 1.0, cy_hole = 1.0; // Centro del buco (coincide con il dominio)

double rho_from_min_angle(double theta_min_deg) {
    if (theta_min_deg <= 0.0 || theta_min_deg >= 180.0) {
        return std::numeric_limits<double>::infinity();
    }

    double theta_rad = theta_min_deg * M_PI / 180.0;
    return 1.0 / (2.0 * std::sin(theta_rad));
}

int main() {
    
    Eigen::Matrix<double, 12, 2> boundary; 
    boundary << 
                //15.0, 0.0,  
                20.0, 0.0,
                //25.0, 0.0,
                30.0, 0.0,
                //35.0, 0.0,
                40.0, 0.0, 
                //40.0, 5.0, 
                40.0, 10.0,
                //40.0, 15.0,
                40.0, 20.0,
                //35.0, 20.0,
                30.0, 20.0,
                //25.0, 20.0,
                20.0, 20.0,
                //15.0, 20.0,
                10.0, 20.0,
                //5.0, 20.0,
                0.0, 20.0,
                //0.0, 15.0,
                0.0, 10.0,
                //0.0, 5.0,
                0.0, 0.0,
                //5.0, 0.0,
                10.0, 0.0;  

    Eigen::Matrix<double, 7, 2> boundary1;
    boundary1 << 0., 10.,
                 10., 10.,
                 20., 10.,
                 20., 15.,
                 20., 20.,
                 10., 20.,
                 0., 20.;
    Eigen::Matrix<double, 7, 2> boundary2;
    boundary2 << 20., 10.,
                 30., 10.,
                 40., 10.,
                 40., 20.,
                 30., 20.,
                 20., 20.,
                 20, 15.;
    Eigen::Matrix<double, 7, 2> boundary3;
    boundary3 << 0., 0.,
                 10., 0.,
                 20., 0.,
                 20., 5.,
                 20., 10.,
                 10., 10., 
                 0., 10.;
    Eigen::Matrix<double, 7,2> boundary4;
    boundary4 << 20., 0.,
                 30.,0.,
                 40., 0.,
                 40., 10.,
                 30., 10.,
                 20., 10.,
                 20, 5.;
   

    
    Eigen::Matrix<double, 13, 2> boundary_concave;  
    boundary_concave << 0.0, 1.3,
                0.5, 1.0, 
                0.3, 0.3,  
                1.0, 0.7,  
                1.0, 0.0, 
                1.85, 0.15, 
                2.0, 0.3,  
                2.0, 1.3,  
                1.0, 1.3,
                1.0, 2.0,
                0.0, 2.0,  
                0.0, 1.7,
                0.0, 1.5;
    Eigen::Matrix<double, 4, 2> hole_concave1;
    hole_concave1 << 0.5, 1.5,
            0.5, 1.7,
            0.3, 1.7,
            0.3, 1.5;
            

    Eigen::Matrix<double, 4, 2> hole_concave2;
    hole_concave2 << 1.3, 1.2,
            1.6, 1.2,
            1.6, 0.5,
            1.3, 0.5;    
    
    Eigen::Matrix<double, 10, 2> boundary_stairs; //scala
    Eigen::Matrix<double, 10, 2> boundary_stairs; //scala
    boundary_stairs <<    
                0.0, 0.0,  
                0.0, 0.0,  
                12.0, 0.0,
                12.0, 2.0, 
                9.0, 2.0,  
                9.0, 4.0,
                6.0, 4.0,  
                6.0, 6.0,
                3.0, 6.0,
                3.0, 8.0,
                0.0, 8.0;
                0.0, 8.0;

    Eigen::Matrix<double, 9, 2> boundary_stairs1;
    boundary_stairs1 <<    
                6.0, 0.0,  
                9.0, 0.0, 
                12.0, 0.0,
                12.0, 2.0, 
                9.0, 2.0,  
                6.0, 2.0,
                0., 2.0,
                0., 0.0, 
                3.0, 0.0;
    Eigen::Matrix<double, 13, 2> boundary_stairs2;
    boundary_stairs2 << 0., 2.,
    6., 2.,
    9., 2.,
    9., 4.,
    8.0, 4.0,
    7.0, 4.0,
    6.0, 4.0,  
    6.0, 6.0,
    3.0, 6.0,
    3.0, 8.0,
    0.0, 8.0,
    0.0, 6.0,
    0.0, 4.0;

    

    Eigen::Matrix<double, 14, 2> internal;
    internal << //1.0, 4.0,
                9.0, 4.0,
                11.0, 4.0,
                //19.0, 1.0,
                //21.0, 1.0,
                29.0, 4.0,
                31.0, 4.0,
                39.0, 4.0,
                //1.0, 16.0,
                9.0, 16.0,
                11.0, 16.0,
                //19.0, 19.0,
                //21.0, 19.0,
                29.0, 16.0,
                31.0, 16.0,
                //39.0, 16.0,
                1.0, 4.0,
                //1.0, 6.0,
                4.0, 9.0,
                4.0, 11.0,
                //1.0, 14.0,
                //1.0, 16.0,
                //1.0, 19.0,
                //39.0, 4.0,
                //39.0, 6.0,
                36.0, 9.0,
                36.0, 11.0;
                //39.0, 14.0,
                //39.0, 16.0,
                //39.0, 19.0;
                
                
          
/*        
    Eigen::Matrix<double, 4, 2> boundary;
    boundary <<  0.0, 0.0,  
                10.0, 0.0,  
                10.0, 10.0,
            //   0.5, 0.5,  
                0.0, 10.0;*/

    Eigen::Matrix<double, 8, 2> boundary_concave_ref;
    boundary_concave_ref <<  0, 0, 
                 4, 0, 
                 4, 2, 
                 2.5, 1.5, 
                 4, 4, 
                 0, 4, 
                 0, 2, 
                 1.5, 2.5;
    Eigen::Matrix<double, 1, 2> int_pt;
    int_pt << 0., 0.;

    Eigen::Matrix<double,10,2> star_points_ref;
    star_points_ref << 
    0.0, 100.0,               // Punto 1: punta superiore
    -22.45, 30.90,        // Punto 2
    -95.11, 30.90,        // Punto 3: punta sinistra
    -36.33, -11.80,       // Punto 4
    -58.78, -80.90,       // Punto 5: punta inferiore sinistra
    0.0, -38.20,           // Punto 6
    58.78, -80.90,        // Punto 7: punta inferiore destra
    36.33, -11.80,        // Punto 8
    95.11, 30.90,         // Punto 9: punta destra
    22.45, 30.90;  

    Eigen::Matrix<double,5,2> star_hole;
    star_hole << 0.0000, 15.00,
    -14.26, 4.64,
    -8.81, -12.36,
    8.81, -12.36,
    14.26, 4.64;


    Eigen::Matrix<double,108,2> C;
    C << 
    -0.910947171536292,-0.160624564341911,
    -0.869215674226965,-0.316368632576244,
    -0.801073498500606,-0.4625,
    -0.708591109885055,-0.594578538960049,
    -0.594578538960049,-0.708591109885055,
    -0.4625,-0.801073498500606,
    -0.316368632576244,-0.869215674226965,
    -0.160624564341911,-0.910947171536292,
    5.6638043864285e-17,-0.925,
    0.166666666666667,-0.925,
    0.333333333333333,-0.925,
    0.5,-0.925,
    0.666666666666667,-0.925,
    0.833333333333333,-0.925,
    1,-0.925,
    1.16666666666667,-0.925,
    1.33333333333333,-0.925,
    1.5,-0.925,
    1.66666666666667,-0.925,
    1.83333333333333,-0.925,
    2,-0.925,
    2.16666666666667,-0.925,
    2.33333333333333,-0.925,
    2.5,-0.925,
    2.66666666666667,-0.925,
    2.83333333333333,-0.925,
    3,-0.925,
    3.16072704159334,-0.89302940365474,
    3.29698484809835,-0.80198484809835,
    3.38802940365474,-0.665727041593338,
    3.42,-0.505,
    3.38802940365474,-0.344272958406662,
    3.29698484809835,-0.20801515190165,
    3.16072704159334,-0.11697059634526,
    3,-0.085,
    2.83333333333333,-0.085,
    2.66666666666667,-0.085,
    2.5,-0.085,
    2.33333333333333,-0.085,
    2.16666666666667,-0.085,
    2,-0.085,
    1.83333333333333,-0.085,
    1.66666666666667,-0.085,
    1.5,-0.085,
    1.33333333333333,-0.085,
    1.16666666666667,-0.085,
    1,-0.085,
    0.833333333333333,-0.085,
    0.666666666666667,-0.085,
    0.5,-0.085,
    0.333333333333333,-0.085,
    0.166666666666667,-0.085,
    5.2045770037451e-18,-0.085,
    -0.085,1.04091540074902e-17,
    5.2045770037451e-18,0.085,
    0.166666666666667,0.085,
    0.333333333333333,0.085,
    0.5,0.085,
    0.666666666666667,0.085,
    0.833333333333333,0.085,
    1,0.085,
    1.16666666666667,0.085,
    1.33333333333333,0.085,
    1.5,0.085,
    1.66666666666667,0.085,
    1.83333333333333,0.085,
    2,0.085,
    2.16666666666667,0.085,
    2.33333333333333,0.085,
    2.5,0.085,
    2.66666666666667,0.085,
    2.83333333333333,0.085,
    3,0.085,
    3.16072704159334,0.11697059634526,
    3.29698484809835,0.20801515190165,
    3.38802940365474,0.344272958406662,
    3.42,0.505,
    3.38802940365474,0.665727041593338,
    3.29698484809835,0.80198484809835,
    3.16072704159334,0.89302940365474,
    3,0.925,
    2.83333333333333,0.925,
    2.66666666666667,0.925,
    2.5,0.925,
    2.33333333333333,0.925,
    2.16666666666667,0.925,
    2,0.925,
    1.83333333333333,0.925,
    1.66666666666667,0.925,
    1.5,0.925,
    1.33333333333333,0.925,
    1.16666666666667,0.925,
    1,0.925,
    0.833333333333333,0.925,
    0.666666666666667,0.925,
    0.5,0.925,
    0.333333333333333,0.925,
    0.166666666666667,0.925,
    5.6638043864285e-17,0.925,
    -0.160624564341911,0.910947171536292,
    -0.316368632576244,0.869215674226965,
    -0.4625,0.801073498500606,
    -0.594578538960049,0.708591109885055,
    -0.708591109885055,0.594578538960049,
    -0.801073498500606,0.4625,
    -0.869215674226965,0.316368632576244,
    -0.910947171536292,0.160624564341911,
    -0.925,1.1327608772857e-16;

Eigen::Matrix<double, 22, 2> skyline;
skyline <<  0.0, 0.0,      // Punto iniziale in basso a sinistra
            0.0, 15.0,     // Edificio 1 sinistro
            5.0, 15.0,
            5.0, 10.0,     // Discesa
            10.0, 10.0,
            10.0, 13.0,    // Edificio più basso
            15.0, 13.0,
            15.0, 18.0,    // Edificio medio
            25.0, 18.0,
            25.0, 20.0,    // Edificio più alto
            26.5, 22.0,
            27.0, 25.0,
            27.45, 30.0,
            28.0, 25.0,
            28.5, 22.0,
            30.0, 20.0,
            30.0, 17.0,    // Scala discendente
            35.0, 17.0,
            35.0, 12.0,
            40.0, 12.0,
            40.0, 0.0,
            25.0,0.0;    // Chiusura a destra     

Eigen::Matrix<double, 10, 2> building1;
building1 <<  0.0, 0.0,      // Punto iniziale in basso a sinistra
            0.0, 15.0,     // Edificio 1 sinistro
            5.0, 15.0,
            5.0, 10.0,     // Discesa
            10.0, 10.0,
            10.0, 13.0,    // Edificio più basso
            15.0, 13.0,
            15.0, 18.0,    // Edificio medio
            25.0, 18.0,
            25.0,0.0;


Eigen::Matrix<double, 15, 2> building2;
building2 <<  25.0, 0.0,
            25.0, 18.0,
            25.0, 20.0,    // Edificio più alto
            26.5, 22.0,
            27.0, 25.0,
            27.4, 30.0,
            27.5, 30.0,
            28.0, 25.0,
            28.5, 22.0,
            30.0, 20.0,
            30.0, 17.0,    // Scala discendente
            35.0, 17.0,
            35.0, 12.0,
            40.0, 12.0,
            40.0, 0.0;  // Chiusura a destra

/*Eigen::Matrix<double, 6, 2> building3;
building3 <<  30.0, 0.0,
            30.0, 17.0,    // Scala discendente
            35.0, 17.0,
            35.0, 12.0,
            40.0, 12.0,
            40.0, 0.0;    // Chiusura a destra */

Eigen::Matrix<double, 4, 2> hole_building1;
hole_building1 << 20., 10.,
        22., 10.,
        22., 12.,
        20., 12.;     

Eigen::Matrix<double, 4, 2> hole_building2;
hole_building2 << 20., 5.,
        22., 5.,
        22., 7.,
        20., 7.;  

std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>>> holes_b(4);
holes_b[0]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {hole_building1, hole_building2};
holes_b[1]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {hole_building1, hole_building2};
holes_b[2]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {};
holes_b[3]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {};


    //------------------------------- REFINEMENT POTENZIATO ESEMPI----------------------------------------------------------------
    
    Eigen::Matrix<double, 5, 2> triang_isoscele;
    triang_isoscele << 
    -10, 0.0,      // B
    //-0.5, 1.,      // M_AB
    -0.7, std::sqrt(3)/2,
     0.0, 2.1,      // A
     2, 1.,      // M_AC
     5., 0.0;      // C
     //-0.55, 0.0;

    Eigen::Matrix<double, 1, 2> int_is;
    int_is << 0.0, 1.0;
             //0.1, 1.5,
             //0.7, 0.5;
            //-0.6, std::sqrt(3)/20;
    
    
    
    Eigen::Matrix<double, 4, 2> hole;
    hole << 10., 5.,
    10., 15.,
    30., 15.,
    30., 5.;
    Eigen::Matrix<double, 8, 2> hole3;
    hole3 << 10., 5.,
    10., 10.,
    10., 15.,
    20., 15.,
    30., 15.,
    30., 10.,
    30., 5.,
    20., 5.;
    Eigen::Matrix<double, 4, 2> hole2;
    hole2 << 3.,4.,
    3., 8.,
    6., 8.,
    6., 4.;
    Eigen::Matrix<double, 5, 2> hole4;
    hole4 << 36.5, 15.,
    37.5, 16.5,
    36.5, 18.,
    38.5, 18.,
    38.5, 15.;
    Eigen::Matrix<double, 5, 2> hole5;
    hole5 << 30.5, 5.,
    32.5, 5.,
    38.5, 5.,
    38.5, 8.,
    30.5, 8.;
    
    
    Eigen::Matrix<double, 1, 2> internal1;
    internal1 << 35., 4.5;
    
    Eigen::Matrix<double, 6, 2> hole_stairs;
    hole_stairs << 4.0, 2.5,   
         5.0, 2.5,   
         5.3, 3.0,   
         5.0, 3.5,   
         4.0, 3.5,   
         3.7, 3.0; 
    Eigen::Matrix<double, 6, 2> heart_hole;
    heart_hole <<  
    1.5, 6.6,     // punta inferiore
    2.1, 7.1,     // lato curvo destro
    1.8, 7.6,     // top destro
    1.5, 7.4,     // vertice superiore centrale
    1.2, 7.6,     // top sinistro
    0.9, 7.1;     // lato curvo sinistro
    
    /*std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>>> holes(5);
    /*std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>>> holes(5);
    holes[0]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {hole4, hole2, hole5};
    holes[1]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {};
    holes[2]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {hole4};
    holes[3]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {hole2};
    holes[4]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {hole5};*/
    std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>>> holes(3);
    holes[4]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {hole5};*/
    std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>>> holes(3);
    holes[0]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {heart_hole};
    holes[1]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {};
    holes[2]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {heart_hole};
    holes[2]= std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {heart_hole};

    //Delaunay<2, 2> del(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {boundary, boundary1, boundary2, boundary3, boundary4}, internal1, holes);  //
    //del.Ruppert_refinement(1.5);  // Esegui il Ruppert refinement con un raggio minimo di 1.0
    //Delaunay<2, 2> del(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {boundary_stairs, boundary_stairs1, boundary_stairs2}, 0, holes);  //
    //del.Ruppert_refinement(1.5);

    //Delaunay<2, 2> del(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {skyline, building1, building2}, 0, holes_b);

    Delaunay<2, 2> del(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> {boundary_stairs}, 20,2);
    //del.print_statistics();
    
///////////TEST OF COMPUTATIONAL EFFICIENCY////////////////


    /*std::ofstream outfile("fdaPDE/src/timing_results.csv");
    outfile << "NumPoints,TimeElapsed(ms)\n";
    
    auto start = high_resolution_clock::now();  // starting measuring time 
    Delaunay<2, 2> del_1000(boundary,1000);  
    //del_10000.Ruppert_refinement(2.0);
    auto end = high_resolution_clock::now();    // ending measuring time 
    auto duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for " << del_1000.dcel().n_nodes() << " points: " << duration << " ms" << std::endl;
    //outfile << del_1000.dcel().n_nodes() << "," << duration << "\n"; 
    outfile << 1000 << "," << duration << "\n"; 

    start = high_resolution_clock::now();  // starting measuring time 
    Delaunay<2, 2> del_10000(boundary,10000);  
    //del_100000.Ruppert_refinement(2.0);
    end = high_resolution_clock::now();    // ending measuring time 
    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for " << del_10000.dcel().n_nodes() << " points: " << duration << " ms" << std::endl;
    //outfile << del_10000.dcel().n_nodes() << "," << duration << "\n"; 
    outfile << 10000 << "," << duration << "\n"; 

    start = high_resolution_clock::now();  // starting measuring time 
    Delaunay<2, 2> del_100000(boundary,100000);  
    //del_100000.Ruppert_refinement(2.0);
    end = high_resolution_clock::now();    // ending measuring time 
    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for " << del_100000.dcel().n_nodes() << " points: " << duration << " ms" << std::endl;
    //outfile << del_100000.dcel().n_nodes() << "," << duration << "\n"; 
    outfile << 100000 << "," << duration << "\n"; */

    /*start = high_resolution_clock::now();  // starting measuring time 
    Delaunay<2, 2> del_100000(boundary,100000);  
    //del_100000.Ruppert_refinement(2.0);
    end = high_resolution_clock::now();    // ending measuring time 
    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for " << 100000 << " points: " << duration << " ms" << std::endl;
    outfile << 100000 << "," << duration << "\n"; 

    start = high_resolution_clock::now();  // starting measuring time 
    Delaunay<2, 2> delaunay_1000000(boundary,1000000);  
    end = high_resolution_clock::now();    // ending measuring time 
    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for " << 1000000 << " points: " << duration << " ms" << std::endl;
    outfile << 1000000 << "," << duration << "\n"; */

    /*auto start = high_resolution_clock::now();  // starting measuring time 
    Delaunay<2, 2> delaunay_8000(boundary,1000, std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> { hole2, hole3});  
    auto end = high_resolution_clock::now();    // ending measuring time 
    auto duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for " << 8000 << " points: " << duration << " ms" << std::endl;
    outfile << 8000 << "," << duration << "\n"; 

    start = high_resolution_clock::now();  // starting measuring time 
    Delaunay<2, 2> delaunay_10000(boundary,5000, std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> { hole2, hole3});  
    end = high_resolution_clock::now();    // ending measuring time 
    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for " << 10000 << " points: " << duration << " ms" << std::endl;
    outfile << 10000 << "," << duration << "\n"; 

    start = high_resolution_clock::now();  // starting measuring time 
    Delaunay<2, 2> delaunay_20000(boundary,10000, std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> { hole2, hole3});  
    end = high_resolution_clock::now();    // ending measuring time 
    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for " << 20000 << " points: " << duration << " ms" << std::endl;
    outfile << 20000 << "," << duration << "\n"; */
    
    //outfile.close();
    
    return 0;
}