#include <fstream>
#include <nlohmann/json.hpp> 
using json = nlohmann::json;
#include <geometry.h>
using namespace fdapde;
#include "domains.h"

int main() {
          
    //-----------------------TEST OF COMPUTATIONAL EFFICIENCY OF REFINEMENT ALGORITHM------------------------------

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> bd={rectangle};

    std::ofstream outfile("Meshes/Delaunay/timing_refinement.csv");
    outfile << "NumPoints,TimeElapsed(ms)\n";
    
    Delaunay<2, 2> del1(bd,0);
    auto start = high_resolution_clock::now();    
    del1.refinement(20, del1.domain_area()/1000.0);
    auto end = high_resolution_clock::now();  

    auto duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for "<<del1.dcel().n_nodes() <<" points: " << duration << " ms" << std::endl; 
    outfile << del1.dcel().n_nodes() << "," << duration << "\n"; 

    Delaunay<2, 2> del2(bd,0);
    start = high_resolution_clock::now();    
    del2.refinement(20, del2.domain_area()/10000.0);
    end = high_resolution_clock::now(); 

    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for "<<del2.dcel().n_nodes() <<" points: " << duration << " ms" << std::endl; 
    outfile << del2.dcel().n_nodes() << "," << duration << "\n"; 

    Delaunay<2, 2> del3(bd,0);
    start = high_resolution_clock::now();    
    del3.refinement(20, del3.domain_area()/100000.0);
    end = high_resolution_clock::now();  

    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for "<<del3.dcel().n_nodes() <<" points: " << duration << " ms" << std::endl; 
    outfile << del3.dcel().n_nodes() << "," << duration << "\n"; 

    outfile.close(); 
    
    return 0;

}